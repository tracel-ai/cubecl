use cubecl::{
    ir::{ElemType, FloatKind},
    prelude::*,
    std::throughput::{measure_launch_overhead, measure_peak_throughput},
    throughput::{CmmaDims, ComputeCmmaConfig, ThroughputKey, ThroughputMode},
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
        #[cfg(feature = "metal-native")]
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

/// Peak memory (copy) throughput.
pub fn memory<R: Runtime>(device: &R::Device) {
    run::<R>(device, &[memory_key()]);
}

/// Measures the fixed cost of a single kernel launch.
pub fn launch_overhead<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let duration = measure_launch_overhead::<R>(&client);
    println!("Launch overhead: {:?}", duration);
}

/// Runs every throughput benchmark and prints them as a table.
pub fn all<R: Runtime>(device: &R::Device) {
    run::<R>(
        device,
        &[compute_direct_key(), compute_cmma_key(), memory_key()],
    );
    launch_overhead::<R>(device);
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
        ThroughputMode::ComputeCmma(cfg) => format!(
            "{}→{} {}×{}×{}",
            key.dtype, cfg.accumulator_type, cfg.cmma_dims.m, cfg.cmma_dims.n, cfg.cmma_dims.k,
        ),
        _ => key.dtype.to_string(),
    }
}

fn mode_label(mode: &ThroughputMode) -> &'static str {
    match mode {
        ThroughputMode::ComputeDirect => "compute-direct",
        ThroughputMode::ComputeCmma(_) => "compute-cmma",
        ThroughputMode::Memory => "memory",
    }
}

fn compute_direct_key() -> ThroughputKey {
    ThroughputKey {
        mode: ThroughputMode::ComputeDirect,
        dtype: ElemType::Float(FloatKind::F16),
    }
}

fn compute_cmma_key() -> ThroughputKey {
    ThroughputKey {
        mode: ThroughputMode::ComputeCmma(ComputeCmmaConfig {
            cmma_dims: CmmaDims {
                m: 16,
                n: 16,
                k: 16,
            },
            accumulator_type: ElemType::Float(FloatKind::F16),
        }),
        dtype: ElemType::Float(FloatKind::F16),
    }
}

fn memory_key() -> ThroughputKey {
    ThroughputKey {
        mode: ThroughputMode::Memory,
        dtype: ElemType::Float(FloatKind::F16),
    }
}
