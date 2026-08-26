use cubecl::{
    ir::{ElemType, FloatKind},
    prelude::*,
    std::throughput::{measure_memory_curve, measure_peak_throughput, measure_resident_curve},
    throughput::{
        CmmaDims, ComputeCmmaConfig, MemoryAccess, MemoryCurve, ThroughputKey, ThroughputMode,
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

/// Peak direct (non-CMMA) compute throughput.
pub fn compute_direct<R: Runtime>(device: &R::Device) {
    run::<R>(device, &[compute_direct_key()]);
}

/// Peak CMMA (tensor-core) compute throughput.
pub fn compute_cmma<R: Runtime>(device: &R::Device) {
    run::<R>(device, &[compute_cmma_key()]);
}

/// Peak memory (copy) throughput, reads and writes both counted.
pub fn memory<R: Runtime>(device: &R::Device) {
    run::<R>(device, &[memory_key()]);
}

/// Peak read-only streaming throughput. Expect this to exceed
/// [`memory`], which pays for a store the read-only case never issues.
pub fn memory_read<R: Runtime>(device: &R::Device) {
    run::<R>(device, &[memory_read_key()]);
}

/// Peak write-only streaming throughput. Expect this to exceed
/// [`memory`], which pays for a read the write-only case never issues.
pub fn memory_write<R: Runtime>(device: &R::Device) {
    run::<R>(device, &[memory_write_key()]);
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
        let cold = measure_memory_curve::<R>(&client, access);
        let resident = measure_resident_curve::<R>(&client, access);
        print_curve(access, &cold, &resident);
    }
}

fn print_curve(access: MemoryAccess, cold: &MemoryCurve, resident: &MemoryCurve) {
    println!("\n  {access:?}{:>16}{:>14}", "cold", "resident");

    for point in cold.points() {
        let rate = |curve: &MemoryCurve| match curve.ceiling_at(point.bytes) {
            Some(bytes_per_s) => format!("{:.1} GB/s", bytes_per_s / 1e9),
            None => String::from("N/A"),
        };

        println!(
            "    {:>10}{:>14}{:>14}",
            bytes_label(point.bytes),
            rate(cold),
            rate(resident),
        );
    }
}

fn bytes_label(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];

    let mut value = bytes as f64;
    let mut unit = 0;

    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }

    format!("{value:.0} {}", UNITS[unit])
}

/// Measures the fixed cost of a single kernel launch.
pub fn launch_overhead<R: Runtime>(device: &R::Device) {
    run::<R>(device, &[launch_overhead_key()]);
}

/// Runs every throughput benchmark and prints them as a table.
pub fn all<R: Runtime>(device: &R::Device) {
    run::<R>(
        device,
        &[
            compute_direct_key(),
            compute_cmma_key(),
            memory_key(),
            memory_read_key(),
            memory_write_key(),
            launch_overhead_key(),
        ],
    );
}

fn run<R: Runtime>(device: &R::Device, keys: &[ThroughputKey]) {
    let client = R::client(device);

    println!("Peak throughput — {}", R::name(&client));
    for &key in keys {
        let value = measure_peak_throughput(&client, key).format(&key);

        println!(
            "  {:<15}{:<24}{:>18}",
            mode_label(&key.mode),
            describe(&key),
            value,
        );
    }
}

/// Describes the operands of a benchmark: input dtype, plus CMMA shape and accumulator.
fn describe(key: &ThroughputKey) -> String {
    match key.mode {
        ThroughputMode::ComputeCmma {
            dtype: input_dtype,
            config: cfg,
        } => format!(
            "{}→{} {}×{}×{}",
            input_dtype, cfg.accumulator_type, cfg.cmma_dims.m, cfg.cmma_dims.n, cfg.cmma_dims.k,
        ),
        ThroughputMode::ComputeDirect { .. } => key.dtype().to_string(),
        ThroughputMode::MemoryWorkingSet { bytes, .. }
        | ThroughputMode::MemoryResident { bytes, .. } => bytes_label(bytes),
        ThroughputMode::Memory
        | ThroughputMode::MemoryRead
        | ThroughputMode::MemoryWrite
        | ThroughputMode::Launch => String::new(),
    }
}

fn mode_label(mode: &ThroughputMode) -> &'static str {
    match mode {
        ThroughputMode::ComputeDirect { .. } => "compute-direct",
        ThroughputMode::ComputeCmma { .. } => "compute-cmma",
        ThroughputMode::Memory
        | ThroughputMode::MemoryWorkingSet {
            access: MemoryAccess::Copy,
            ..
        }
        | ThroughputMode::MemoryResident {
            access: MemoryAccess::Copy,
            ..
        } => "memory",
        ThroughputMode::MemoryRead
        | ThroughputMode::MemoryWorkingSet {
            access: MemoryAccess::Read,
            ..
        }
        | ThroughputMode::MemoryResident {
            access: MemoryAccess::Read,
            ..
        } => "memory-read",
        ThroughputMode::MemoryWrite
        | ThroughputMode::MemoryWorkingSet {
            access: MemoryAccess::Write,
            ..
        }
        | ThroughputMode::MemoryResident {
            access: MemoryAccess::Write,
            ..
        } => "memory-write",
        ThroughputMode::Launch => "launch",
    }
}

fn compute_direct_key() -> ThroughputKey {
    ThroughputKey {
        mode: ThroughputMode::ComputeDirect {
            dtype: ElemType::Float(FloatKind::F16),
        },
    }
}

fn compute_cmma_key() -> ThroughputKey {
    ThroughputKey {
        mode: ThroughputMode::ComputeCmma {
            dtype: ElemType::Float(FloatKind::F16),
            config: ComputeCmmaConfig {
                cmma_dims: CmmaDims {
                    m: 16,
                    n: 16,
                    k: 16,
                },
                accumulator_type: ElemType::Float(FloatKind::F16),
            },
        },
    }
}

fn memory_key() -> ThroughputKey {
    ThroughputKey {
        mode: ThroughputMode::Memory,
    }
}

fn memory_read_key() -> ThroughputKey {
    ThroughputKey {
        mode: ThroughputMode::MemoryRead,
    }
}

fn memory_write_key() -> ThroughputKey {
    ThroughputKey {
        mode: ThroughputMode::MemoryWrite,
    }
}

fn launch_overhead_key() -> ThroughputKey {
    ThroughputKey {
        mode: ThroughputMode::Launch,
    }
}
