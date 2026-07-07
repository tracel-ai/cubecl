use cubecl_core::ir::ElemType;
use cubecl_runtime::{
    client::ComputeClient,
    runtime::Runtime,
    throughput::{ThroughputKey, ThroughputMode, ThroughputValue},
};

use crate::throughput::{compute_cmma, compute_direct, memory_direct};

pub fn peak_throughput<R: Runtime>(
    client: &ComputeClient<R>,
    key: ThroughputKey,
) -> ThroughputValue {
    let launch_config = launch_config(client, key.dtype);

    let kernel_config = match key.mode {
        ThroughputMode::ComputeDirect => compute_direct::build_kernel(client, key, launch_config),
        ThroughputMode::ComputeCmma(cmma_config) => {
            compute_cmma::build_kernel(client, key, cmma_config, launch_config)
        }
        ThroughputMode::Memory => memory_direct::build_kernel(client, key, launch_config),
    };

    client.throughput(key, kernel_config)
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

    let plane = hardware.plane_size_max.max(1);
    let cube_dim = (hardware.max_units_per_cube.min(256) / plane * plane)
        .max(plane)
        .min(hardware.max_cube_dim.0);

    let sms = hardware.num_streaming_multiprocessors.unwrap_or(64);
    let cube_count = (sms * 32).min(hardware.max_cube_count.0);

    let vector_size = client
        .io_optimized_vector_sizes(dtype.size())
        .next()
        .unwrap_or(1);

    let plane_size = client.properties().hardware.plane_size_max.max(1) as usize;

    LaunchConfig {
        cube_dim: cube_dim as usize,
        cube_count: cube_count as usize,
        vector_size,
        plane_size,
    }
}
