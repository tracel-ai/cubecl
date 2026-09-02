use cubecl::{
    ir::{ElemType, FloatKind},
    prelude::*,
    std::throughput::{measure_memory_curve, measure_peak_throughput},
    throughput::{
        CmmaDims, ComputeCmmaConfig, MemoryAccess, MemoryCurve, ThroughputError, ThroughputKey,
        ThroughputMode,
    },
};

/// Binds the runtime selected by the enabled cargo feature to a type alias and runs `$body`.
///
/// Keeps backend selection in one place so binaries don't each repeat the `cfg` block:
/// `dispatch!(R => throughput::compute_direct::<R>(&Default::default()))`.
#[macro_export]
macro_rules! dispatch {
    ($runtime:ident => $body:expr) => {{
        #[cfg(feature = "cuda")]
        {
            type $runtime = cubecl::cuda::CudaRuntime;
            $body;
        }
        #[cfg(feature = "hip")]
        {
            type $runtime = cubecl::hip::HipRuntime;
            $body;
        }
        #[cfg(feature = "cpu")]
        {
            type $runtime = cubecl::cpu::CpuRuntime;
            $body;
        }
        #[cfg(all(feature = "metal-native", target_vendor = "apple"))]
        {
            type $runtime = cubecl::metal::MetalRuntime;
            $body;
        }
        // All wgpu sub-backends (WGSL, Vulkan/SPIR-V, Metal/MSL, WebGPU) share `WgpuRuntime`;
        // the compiler is chosen by the enabled `cubecl` sub-feature and the adapter.
        #[cfg(feature = "wgpu")]
        {
            type $runtime = cubecl::wgpu::WgpuRuntime;
            $body;
        }
    }};
}

/// Peak arithmetic throughput, per float type the device supports.
pub fn compute_direct<R: Runtime>(device: &R::Device) {
    report::<R>(device, compute_direct_rows);
}

/// Peak cooperative-matrix throughput, per accumulator width.
pub fn compute_cmma<R: Runtime>(device: &R::Device) {
    report::<R>(device, compute_cmma_rows);
}

/// Peak memory (copy) throughput, reads and writes both counted.
pub fn memory<R: Runtime>(device: &R::Device) {
    report::<R>(device, |_| vec![memory_row(MemoryAccess::Copy)]);
}

/// Peak read-only streaming throughput. Expect this to exceed
/// [`memory`], which pays for a store the read-only case never issues.
pub fn memory_read<R: Runtime>(device: &R::Device) {
    report::<R>(device, |_| vec![memory_row(MemoryAccess::Read)]);
}

/// Peak write-only streaming throughput. Expect this to exceed
/// [`memory`], which pays for a read the write-only case never issues.
pub fn memory_write<R: Runtime>(device: &R::Device) {
    report::<R>(device, |_| vec![memory_row(MemoryAccess::Write)]);
}

/// Peak memory throughput as a function of working set size, for both access
/// patterns.
///
/// The single-size probes above report the last row of each table; the rows
/// above it are what a kernel moving that much can actually hit.
pub fn memory_curve<R: Runtime>(device: &R::Device) {
    let client = R::client(device);

    println!("Memory curve — {}", R::name(&client));

    for access in [MemoryAccess::Read, MemoryAccess::Write, MemoryAccess::Copy] {
        print_curve(access, &measure_memory_curve::<R>(&client, access));
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
pub fn launch_overhead<R: Runtime>(device: &R::Device) {
    report::<R>(device, |_| vec![launch_row()]);
}

/// Runs every throughput benchmark and prints them as a table.
pub fn all<R: Runtime>(device: &R::Device) {
    report::<R>(device, |client| {
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

fn report<R: Runtime>(device: &R::Device, rows: impl FnOnce(&ComputeClient<R>) -> Vec<Row>) {
    let client = R::client(device);

    println!(
        "Peak throughput — {} / {}",
        R::name(&client),
        client.properties().identity.name
    );

    for row in rows(&client) {
        let value = match row.key {
            Some(key) => match measure_peak_throughput(&client, key) {
                Ok(value) => value.format(&key),
                Err(unavailable) => unavailable.to_string(),
            },
            // A row the device could not even be asked for: no tile to name, no
            // type to compute in.
            None => ThroughputError::Unsupported.to_string(),
        };

        println!("  {:<15}{:<24}{:>18}", row.mode, row.operands, value);
    }
}

fn compute_direct_rows<R: Runtime>(client: &ComputeClient<R>) -> Vec<Row> {
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
fn compute_cmma_rows<R: Runtime>(client: &ComputeClient<R>) -> Vec<Row> {
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
fn largest_cmma<R: Runtime>(
    client: &ComputeClient<R>,
    dtype: ElemType,
    accumulator_type: ElemType,
) -> Option<CmmaDims> {
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
