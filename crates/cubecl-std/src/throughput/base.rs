use cubecl_core::ir::ElemType;
use cubecl_runtime::{
    client::ComputeClient,
    runtime::Runtime,
    throughput::{LaunchConfig, ThroughputKey, ThroughputMode, ThroughputRunner, ThroughputValue},
};

use crate::throughput::{ComputeCmmaRunner, ComputeDirectRunner, MemoryDirectRunner};

pub fn peak_throughput<R: Runtime>(
    client: &ComputeClient<R>,
    key: ThroughputKey,
) -> ThroughputValue {
    let launch_config = launch_config(client, key.dtype);

    let kernel_config = match key.mode {
        ThroughputMode::ComputeDirect => {
            <ComputeDirectRunner as ThroughputRunner<R>>::build_kernel(
                client,
                key.dtype,
                launch_config,
            )
        }
        ThroughputMode::ComputeCmma => <ComputeCmmaRunner as ThroughputRunner<R>>::build_kernel(
            client,
            key.dtype,
            launch_config,
        ),
        ThroughputMode::Memory => <MemoryDirectRunner as ThroughputRunner<R>>::build_kernel(
            client,
            key.dtype,
            launch_config,
        ),
    };

    client.throughput(key, kernel_config)
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

    LaunchConfig {
        cube_dim: cube_dim as usize,
        cube_count: cube_count as usize,
        vector_size,
    }
}
