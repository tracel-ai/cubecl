use cubecl::{
    Device,
    ir::{ElemType, FloatKind},
    prelude::Client,
    std::throughput::{measure_memory_curve, measure_peak_throughput},
    throughput::{
        CmmaDims, ComputeCmmaConfig, MemoryAccess, MemoryCurve, ThroughputError, ThroughputKey,
        ThroughputMode,
    },
};
use cubecl_dispatch::DeviceExt;

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
