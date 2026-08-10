use cubecl_core::ir::ElemType;
use cubecl_runtime::{
    client::ComputeClient,
    runtime::Runtime,
    server::CubeDim,
    throughput::{ThroughputKey, ThroughputMode, ThroughputValue},
    tune::{Bounds, Thresholds, Work, calculate_bounds},
};

use crate::throughput::{compute_cmma, compute_direct, launch_overhead, memory_direct};

/// Measure peak throughput on `device` for each of the given `keys`.
pub fn device_throughput<R: Runtime>(
    device: &R::Device,
    keys: &[ThroughputKey],
) -> alloc::vec::Vec<ThroughputValue> {
    let client = R::client(device);
    keys.iter()
        .map(|key| measure_peak_throughput::<R>(&client, *key))
        .collect()
}

/// Computes the peak throughput for a given runtime and key.
///
/// Native only, panics on WASM
pub fn measure_peak_throughput<R: Runtime>(
    client: &ComputeClient<R>,
    key: ThroughputKey,
) -> ThroughputValue {
    // A throughput probe is a measurement: inside a dry run its launches must
    // still execute — a skipped launch would be timed anyway and cache a
    // garbage peak in the device-level throughput store — and the buffers it
    // allocates are probe scratch, not part of the workload's memory plan.
    let _measurement = cubecl_runtime::dry_run::RealRun::new();

    let launch_config = launch_config(client, key.dtype());

    let kernel_config = match key.mode {
        ThroughputMode::ComputeDirect { .. } => {
            compute_direct::build_kernel(client, key, launch_config)
        }
        ThroughputMode::ComputeCmma {
            config: cmma_config,
            ..
        } => {
            if client.properties().features.matmul.cmma.is_empty() {
                return ThroughputValue::ZERO;
            }
            compute_cmma::build_kernel(client, key, cmma_config, launch_config)
        }
        ThroughputMode::Memory => memory_direct::build_kernel(client, key, launch_config),
        ThroughputMode::Launch => launch_overhead::build_kernel(client, key, launch_config),
    };

    let value = client.measure_throughput(key, kernel_config);

    client.memory_cleanup();

    value
}

/// Calculates roofline autotune bounds for a given [`Work`] amount and compute throughput key.
///
/// Measures compute and memory peak throughputs along with launch overhead for the runtime client.
pub fn roofline_bounds<R: Runtime>(
    client: &ComputeClient<R>,
    compute_key: ThroughputKey,
    work: Work,
    thresholds: Thresholds,
) -> Bounds {
    let memory_key = ThroughputKey {
        mode: ThroughputMode::Memory,
    };
    let launch_key = ThroughputKey {
        mode: ThroughputMode::Launch,
    };

    Bounds {
        bounds: calculate_bounds(
            work,
            thresholds,
            &measure_peak_throughput(client, compute_key),
            &measure_peak_throughput(client, memory_key),
            &memory_key,
        ),
        launch_overhead: measure_peak_throughput(client, launch_key).duration_per_op(),
    }
}

/// Hardware execution parameters for launching a compute kernel.
#[derive(Clone, Copy)]
pub struct LaunchConfig {
    /// The number of threads per cube.
    pub cube_dim: usize,
    /// The total number of cubes to dispatch.
    pub cube_count: usize,
    /// The vectorization factor (e.g., 4 for `vec4` operations).
    pub vector_size: usize,
    /// The number of threads in a hardware execution plane.
    pub plane_size: usize,
}

fn launch_config<R: Runtime>(client: &ComputeClient<R>, dtype: ElemType) -> LaunchConfig {
    let hardware = &client.properties().hardware;

    let plane_size = hardware.plane_size_max.max(1);
    let requested = (hardware.max_units_per_cube / plane_size * plane_size)
        .max(plane_size)
        .min(hardware.max_cube_dim.0);

    let cube_dim = CubeDim::new(client, requested as usize).num_elems();

    let sms = hardware.num_streaming_multiprocessors.unwrap_or(64);
    let cube_count = (sms * 32).min(hardware.max_cube_count.0);

    let vector_size = client
        .io_optimized_vector_sizes(dtype.size())
        .next()
        .unwrap_or(1);

    LaunchConfig {
        cube_dim: cube_dim as usize,
        cube_count: cube_count as usize,
        vector_size,
        plane_size: plane_size as usize,
    }
}
