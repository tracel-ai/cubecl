use cubecl::{
    Device,
    ir::{ElemType, FloatKind},
    prelude::*,
    std::throughput::{measure_memory_curve, measure_peak_throughput, roofline_bounds},
    throughput::{
        CmmaDims, ComputeCmmaConfig, MemoryAccess, MemoryCurve, ThroughputError, ThroughputKey,
        ThroughputMode,
    },
    tune::{Thresholds, Work},
};

/// Binds the default device of each runtime selected by the enabled cargo features to
/// `$device` and runs `$body` on it.
///
/// Keeps backend selection in one place so binaries don't each repeat the `cfg` block:
/// `dispatch!(device => throughput::compute_direct(&device))`.
#[macro_export]
macro_rules! dispatch {
    ($device:ident => $body:expr) => {{
        #[cfg(feature = "cuda")]
        {
            let $device = cubecl::Device::Cuda(Default::default());
            $body;
        }
        #[cfg(feature = "hip")]
        {
            let $device = cubecl::Device::Hip(Default::default());
            $body;
        }
        #[cfg(feature = "cpu")]
        {
            let $device = cubecl::Device::Cpu(Default::default());
            $body;
        }
        #[cfg(all(feature = "metal-native", target_vendor = "apple"))]
        {
            let $device = cubecl::Device::Metal(Default::default());
            $body;
        }
        // All wgpu sub-backends (WGSL, Vulkan/SPIR-V, Metal/MSL, WebGPU) share `WgpuRuntime`;
        // the compiler is chosen by the enabled `cubecl` sub-feature and the adapter.
        #[cfg(feature = "wgpu")]
        {
            let $device = cubecl::Device::Wgpu(Default::default());
            $body;
        }
    }};
}

/// Peak arithmetic throughput, per float type the device supports.
pub fn compute_direct(device: &Device) {
    report(device, compute_direct_rows);
}

/// Peak cooperative-matrix throughput, per accumulator width.
pub fn compute_cmma(device: &Device) {
    report(device, compute_cmma_rows);
}

/// Peak memory (copy) throughput, reads and writes both counted.
pub fn memory(device: &Device) {
    report(device, |_| vec![memory_row(MemoryAccess::Copy)]);
}

/// Peak read-only streaming throughput. Expect this to exceed
/// [`memory`], which pays for a store the read-only case never issues.
pub fn memory_read(device: &Device) {
    report(device, |_| vec![memory_row(MemoryAccess::Read)]);
}

/// Peak write-only streaming throughput. Expect this to exceed
/// [`memory`], which pays for a read the write-only case never issues.
pub fn memory_write(device: &Device) {
    report(device, |_| vec![memory_row(MemoryAccess::Write)]);
}

/// Peak memory throughput as a function of working set size, for both access
/// patterns.
///
/// The single-size probes above report the last row of each table; the rows
/// above it are what a kernel moving that much can actually hit.
pub fn memory_curve(device: &Device) {
    let client = device.client();

    println!("Memory curve — {}", client.name());

    for access in [MemoryAccess::Read, MemoryAccess::Write, MemoryAccess::Copy] {
        print_curve(access, &measure_memory_curve(&client, access));
    }
}

fn print_curve(access: MemoryAccess, curve: &MemoryCurve) {
    println!("\n  {:<8}{:>18}", format!("{access:?}"), "peak");

    for point in curve.points() {
        let rate = match curve.ceiling_at(point.bytes) {
            Some(bytes_per_s) => format!("{:.1} GB/s", bytes_per_s / 1e9),
            None => String::from("N/A"),
        };

        println!("    {:>10}{:>14}", bytes_label(point.bytes), rate);
    }
}

/// Measures the fixed cost of a single kernel launch.
pub fn launch_overhead(device: &Device) {
    report(device, |_| vec![launch_row()]);
}

/// Runs every throughput benchmark and prints them as a table.
pub fn all(device: &Device) {
    report(device, |client| {
        let mut rows = compute_direct_rows(client);
        rows.extend(compute_cmma_rows(client));
        rows.extend([MemoryAccess::Copy, MemoryAccess::Read, MemoryAccess::Write].map(memory_row));
        rows.push(launch_row());
        rows
    });
}

/// One line of the report, or `None` where the device implements no such thing.
struct Row {
    mode: &'static str,
    operands: String,
    key: Option<ThroughputKey>,
}

fn report(device: &Device, rows: impl FnOnce(&Client) -> Vec<Row>) {
    let client = device.client();
    let start = std::time::Instant::now();

    println!(
        "Peak throughput — {} / {}",
        client.name(),
        client.properties().identity.name
    );

    for row in rows(&client) {
        let value = match row.key {
            Some(key) => match measure_peak_throughput(&client, key) {
                Ok(value) => value.format(&key),
                Err(unavailable) => unavailable.to_string(),
            },
            None => ThroughputError::Unsupported.to_string(),
        };

        println!("  {:<15}{:<24}{:>18}", row.mode, row.operands, value);
    }

    println!("\n  measured in {:.1} s", start.elapsed().as_secs_f64());
}

fn compute_direct_rows(client: &Client) -> Vec<Row> {
    [FloatKind::F32, FloatKind::F16, FloatKind::BF16]
        .into_iter()
        .map(|kind| {
            let dtype = ElemType::Float(kind);
            let supported = client.properties().features.supports_type(dtype);

            Row {
                mode: "compute-direct",
                operands: dtype.to_string(),
                key: supported.then_some(ThroughputKey {
                    mode: ThroughputMode::ComputeDirect { dtype },
                }),
            }
        })
        .collect()
}

/// A row per accumulator width, at f16 inputs.
///
/// Consumer parts halve their tensor rate for f32 accumulation, which is the
/// one a matmul runs on.
fn compute_cmma_rows(client: &Client) -> Vec<Row> {
    let dtype = ElemType::Float(FloatKind::F16);

    [FloatKind::F16, FloatKind::F32]
        .into_iter()
        .map(|kind| {
            let accumulator_type = ElemType::Float(kind);
            let dims = largest_cmma(client, dtype, accumulator_type);

            Row {
                mode: "compute-cmma",
                operands: match dims {
                    Some(dims) => {
                        format!(
                            "{dtype}→{accumulator_type} {}×{}×{}",
                            dims.m, dims.n, dims.k
                        )
                    }
                    None => format!("{dtype}→{accumulator_type}"),
                },
                key: dims.map(|cmma_dims| ThroughputKey {
                    mode: ThroughputMode::ComputeCmma {
                        dtype,
                        config: ComputeCmmaConfig {
                            cmma_dims,
                            accumulator_type,
                        },
                    },
                }),
            }
        })
        .collect()
}

/// The largest cooperative matrix the device implements for these operands.
///
/// Read from `cmma` rather than through `select_cmma_tile`, which answers from
/// `mma` as well: a shape only that instruction has does not run here.
fn largest_cmma(client: &Client, dtype: ElemType, accumulator_type: ElemType) -> Option<CmmaDims> {
    client
        .properties()
        .features
        .matmul
        .cmma
        .iter()
        .filter(|it| it.a_type == dtype && it.b_type == dtype && it.cd_type == accumulator_type)
        .max_by_key(|it| it.m as u64 * it.n as u64 * it.k as u64)
        .map(|it| CmmaDims {
            m: it.m as usize,
            n: it.n as usize,
            k: it.k as usize,
        })
}

fn memory_row(access: MemoryAccess) -> Row {
    Row {
        mode: "memory",
        operands: format!(
            "{:<8}{}",
            format!("{access:?}").to_lowercase(),
            bytes_label(access.default_working_set())
        ),
        key: Some(ThroughputKey {
            mode: ThroughputMode::memory(access),
        }),
    }
}

fn launch_row() -> Row {
    Row {
        mode: "launch",
        operands: String::new(),
        key: Some(ThroughputKey {
            mode: ThroughputMode::Launch,
        }),
    }
}

fn bytes_label(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];

    let mut value = bytes as f64;
    let mut unit = 0;

    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }

    format!("{value:.0} {}", UNITS[unit])
}

/// The memory peak measured with the device to itself, then again while other
/// threads probe it.
///
/// A probe that shares the device reports that device's share, so the two
/// numbers agree only if a probe holds the device while it measures. The wall
/// clock is the price of holding it.
pub fn contended(device: &Device) {
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    const LOAD_THREADS: usize = 4;
    const SAMPLES: usize = 5;

    let client = device.client();
    let key = ThroughputKey {
        mode: ThroughputMode::memory(MemoryAccess::Copy),
    };

    println!(
        "Contended probe — {} / {}",
        client.name(),
        client.properties().identity.name
    );

    let quiet = peaks(&client, key, SAMPLES);

    let stop = Arc::new(AtomicBool::new(false));
    let load: Vec<_> = (0..LOAD_THREADS)
        .map(|_| {
            let (client, stop) = (client.clone(), stop.clone());

            std::thread::spawn(move || {
                let key = ThroughputKey {
                    mode: ThroughputMode::memory(MemoryAccess::Read),
                };
                let mut ran = 0;

                while !stop.load(Ordering::Relaxed) {
                    if measure_peak_throughput(&client, key).is_ok() {
                        ran += 1;
                    }
                }

                ran
            })
        })
        .collect();

    let loaded = peaks(&client, key, SAMPLES);

    stop.store(true, Ordering::Relaxed);
    let offered: usize = load
        .into_iter()
        .map(|thread| thread.join().expect("a load thread finishes"))
        .sum();

    println!(
        "  {:<20}{:>12}{:>12}{:>12}{:>10}",
        "", "best", "worst", "wall", "samples"
    );
    report_peaks("quiet", &quiet);
    report_peaks(&format!("under {LOAD_THREADS} threads"), &loaded);
    println!("\n  {offered} probes completed on the load threads meanwhile.");
}

/// `samples` measurements of `key`, in bytes per second, timed end to end.
///
/// Every one of them probes: the run needs `CUBECL_THROUGHPUT_CACHE=0`, or the
/// first answer is served to the rest and nothing is measured under load.
fn peaks(client: &Client, key: ThroughputKey, samples: usize) -> (Vec<f64>, std::time::Duration) {
    let start = std::time::Instant::now();

    let rates = (0..samples)
        .filter_map(|_| {
            measure_peak_throughput(client, key)
                .ok()
                .map(|value| value.bytes_per_s(&key))
        })
        .collect();

    (rates, start.elapsed())
}

fn report_peaks(label: &str, (rates, elapsed): &(Vec<f64>, std::time::Duration)) {
    let best = rates.iter().copied().fold(f64::MIN, f64::max);
    let worst = rates.iter().copied().fold(f64::MAX, f64::min);

    println!(
        "  {label:<20}{:>12}{:>12}{:>12}{:>10}",
        format!("{:.1} GB/s", best / 1e9),
        format!("{:.1} GB/s", worst / 1e9),
        format!("{:.1} s", elapsed.as_secs_f64()),
        rates.len(),
    );
}

#[cube(launch_unchecked)]
fn copy_array<F: Float, N: Size>(input: &[Vector<F, N>], output: &mut [Vector<F, N>]) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS];
    }
}

/// What a probe costs the work already running beside it.
///
/// [`contended`] measures probes against probes, which both hold the device
/// once a probe takes it. Ordinary kernels take nothing, so they are what a
/// held device actually delays: the same launches are timed alone, then again
/// while the probes run.
pub fn pressure(device: &Device) {
    const LOAD_THREADS: usize = 4;
    const LAUNCHES: usize = 4000;
    const PROBES: usize = 5;

    let client = device.client();
    let key = ThroughputKey {
        mode: ThroughputMode::memory(MemoryAccess::Copy),
    };

    println!(
        "Probe pressure — {} / {}",
        client.name(),
        client.properties().identity.name
    );

    let alone = load(&client, LOAD_THREADS, LAUNCHES, || ());
    let (beside, probes) = {
        let probing = std::cell::Cell::new((Vec::new(), std::time::Duration::ZERO));

        let beside = load(&client, LOAD_THREADS, LAUNCHES, || {
            probing.set(peaks(&client, key, PROBES));
        });

        (beside, probing.take())
    };

    println!(
        "  {:<20}{:>12}{:>12}{:>10}",
        "", "launches/s", "wall", "samples"
    );
    report_load("alone", LOAD_THREADS * LAUNCHES, alone);
    report_load("beside the probes", LOAD_THREADS * LAUNCHES, beside);

    println!();
    report_peaks("the probes", &probes);
}

/// Runs `launches` copies per thread on `threads` threads, with `meanwhile` on
/// this one, and reports how long they all took.
fn load(
    client: &Client,
    threads: usize,
    launches: usize,
    meanwhile: impl FnOnce(),
) -> std::time::Duration {
    const BYTES: usize = 64 << 20;

    let start = std::time::Instant::now();

    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| scope.spawn(|| copies(client, BYTES, launches)))
            .collect();

        meanwhile();

        for worker in workers {
            worker.join().expect("a load thread finishes");
        }
    });

    start.elapsed()
}

/// `launches` copies of a `bytes` buffer, resynchronised every round so the
/// loop queues work rather than outrunning the device's task channel.
fn copies(client: &Client, bytes: usize, launches: usize) {
    const LINE: usize = 4;
    const UNITS: u32 = 256;

    let lines = bytes / (size_of::<f32>() * LINE);
    let input = client.empty(bytes);
    let output = client.empty(bytes);

    for _ in 0..launches {
        unsafe {
            copy_array::launch_unchecked::<f32>(
                client,
                CubeCount::Static(lines as u32 / UNITS, 1, 1),
                CubeDim::new_1d(UNITS),
                LINE,
                BufferArg::from_raw_parts(input.clone(), lines),
                BufferArg::from_raw_parts(output.clone(), lines),
            )
        };
    }

    cubecl::future::block_on(client.sync()).expect("the copies run");
}

fn report_load(label: &str, launches: usize, elapsed: std::time::Duration) {
    println!(
        "  {label:<20}{:>12}{:>12}{:>10}",
        format!("{:.0}", launches as f64 / elapsed.as_secs_f64()),
        format!("{:.1} s", elapsed.as_secs_f64()),
        launches,
    );
}

/// Two threads asking for the same peak at once, from a cold cache.
///
/// The cache is read when a probe starts and written when it ends, with no
/// mark in between, so the second thread misses and measures a value the store
/// then declines. Both wall clocks are a whole probe's, against the third
/// call's, which is the cache answering.
pub fn duplicate(device: &Device) {
    let client = device.client();
    let key = ThroughputKey {
        mode: ThroughputMode::ComputeDirect {
            dtype: ElemType::Float(FloatKind::F32),
        },
    };

    println!(
        "Duplicate probe — {} / {}",
        client.name(),
        client.properties().identity.name
    );

    let (first, second) = std::thread::scope(|scope| {
        let first = scope.spawn(|| timed(&client, key));
        let second = scope.spawn(|| timed(&client, key));

        (
            first.join().expect("the first finishes"),
            second.join().expect("the second finishes"),
        )
    });

    println!("  {:<20}{:>12}", "", "wall");
    for (label, elapsed) in [
        ("first", first),
        ("second", second),
        ("after both", timed(&client, key)),
    ] {
        println!(
            "  {label:<20}{:>12}",
            format!("{:.2} s", elapsed.as_secs_f64())
        );
    }
}

fn timed(client: &Client, key: ThroughputKey) -> std::time::Duration {
    let start = std::time::Instant::now();
    let _ = measure_peak_throughput(client, key);

    start.elapsed()
}

/// What one autotune key pays for its bounds, cold and then cached.
///
/// [`roofline_bounds`] measures a compute peak, a memory peak at the work's own
/// footprint, and the launch overhead. Only the memory key carries the
/// footprint, so a key over a different size pays for one more probe while the
/// other two answer from the cache.
pub fn bounds(device: &Device) {
    let client = device.client();
    let compute_key = ThroughputKey {
        mode: ThroughputMode::ComputeDirect {
            dtype: ElemType::Float(FloatKind::F32),
        },
    };

    println!(
        "Autotune bounds — {} / {}",
        client.name(),
        client.properties().identity.name
    );
    println!("  {:<28}{:>12}", "", "wall");

    for (label, bytes) in [
        ("first key", 64 << 20),
        ("same footprint again", 64 << 20),
        ("another footprint", 256 << 20),
        ("a third", 1 << 20),
    ] {
        let work = Work {
            compute_ops: bytes * 4,
            bytes,
        };

        let start = std::time::Instant::now();
        let _ = roofline_bounds(&client, compute_key, work, Thresholds::default());

        println!(
            "  {label:<28}{:>12}",
            format!("{:.2} s", start.elapsed().as_secs_f64())
        );
    }
}
