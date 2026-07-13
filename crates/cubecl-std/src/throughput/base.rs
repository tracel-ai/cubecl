use cubecl_core::{
    CubeCount,
    frontend::BufferArg,
    future::block_on,
    ir::{ElemType, IntKind},
};
use cubecl_runtime::{
    client::ComputeClient,
    runtime::Runtime,
    server::CubeDim,
    throughput::{ThroughputKey, ThroughputMode, ThroughputValue},
};

use crate::throughput::{compute_cmma, compute_direct, launch_overhead, memory_direct};

/// Computes the peak throughput for a given runtime and key.
pub fn measure_peak_throughput<R: Runtime>(
    client: &ComputeClient<R>,
    key: ThroughputKey,
) -> ThroughputValue {
    let launch_config = launch_config(client, key.dtype);

    let kernel_config = match key.mode {
        ThroughputMode::ComputeDirect => compute_direct::build_kernel(client, key, launch_config),
        ThroughputMode::ComputeCmma(cmma_config) => {
            if client.properties().features.matmul.cmma.is_empty() {
                return ThroughputValue::ZERO;
            }
            compute_cmma::build_kernel(client, key, cmma_config, launch_config)
        }
        ThroughputMode::Memory => memory_direct::build_kernel(client, key, launch_config),
    };

    client.measure_throughput(key, kernel_config)
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

/// Measures the fixed cost of a single kernel launch.
pub fn measure_launch_overhead<R: Runtime>(client: &ComputeClient<R>) -> core::time::Duration {
    client.measure_launch_overhead(|| {
        let input = client.empty(size_of::<i32>());
        let output = client.empty(size_of::<i32>());

        let (_, duration) = client
            .profile(
                || unsafe {
                    launch_overhead::launch_overhead::launch_unchecked::<R>(
                        client,
                        CubeCount::new_single(),
                        CubeDim::new_single(),
                        1,
                        BufferArg::from_raw_parts(input.clone(), 1),
                        BufferArg::from_raw_parts(output.clone(), 1),
                        ElemType::Int(IntKind::I32).into(),
                    );
                },
                "launch_overhead",
            )
            .expect("should succeed launch_overhead");

        block_on(duration.into_future()).duration()
    })
}
