use cubecl_core::{
    frontend::{BufferArg, BufferBinding},
    ir::ElemType,
};

use cubecl_runtime::{
    client::ComputeClient,
    runtime::Runtime,
    server::{CubeCount, CubeDim},
    throughput::{ThroughputKey, ThroughputMode, ThroughputValue},
};

use crate::throughput::{
    compute_cmma_throughput, compute_direct_throughput, memory_direct_throughput,
};

use alloc::boxed::Box;

pub fn peak_throughput<R: Runtime>(
    client: &ComputeClient<R>,
    key: ThroughputKey,
) -> ThroughputValue {
    let mode = key.mode;
    let dtype = key.dtype;

    let launch_config = match mode {
        ThroughputMode::ComputeDirect | ThroughputMode::ComputeCmma | ThroughputMode::Memory => {
            compute_launch(client, dtype)
        }
    };

    let kernel = match mode {
        ThroughputMode::ComputeDirect => compute_direct(client, dtype, launch_config, 8, 1024),
        ThroughputMode::ComputeCmma => compute_cmma(client, dtype, launch_config, 8, 1024),
        ThroughputMode::Memory => memory_direct(client, dtype, launch_config, 8),
    };

    client.throughput(key, kernel.unit_count, kernel.kernel)
}

#[derive(Clone, Copy)]
struct LaunchConfig {
    cube_dim: usize,
    cube_count: usize,
    vector_size: usize,
}

fn compute_launch<R: Runtime>(client: &ComputeClient<R>, dtype: ElemType) -> LaunchConfig {
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

pub struct KernelConfig {
    kernel: Box<dyn Fn()>,
    unit_count: usize,
}

fn compute_direct<R: Runtime>(
    client: &ComputeClient<R>,
    dtype: ElemType,
    config: LaunchConfig,
    n_acc: usize,
    n_iter: usize,
) -> KernelConfig {
    let client = client.clone();

    let kernel = Box::new(move || unsafe {
        let out = client.empty(config.vector_size * dtype.size());

        compute_direct_throughput::launch_unchecked(
            &client,
            CubeCount::Static(config.cube_count as u32, 1, 1),
            CubeDim::new_1d(config.cube_dim as u32),
            config.vector_size,
            BufferArg::from_raw_parts(out, 1),
            n_acc,
            n_iter,
            dtype.into(),
        )
    });
    KernelConfig {
        kernel,
        unit_count: compute_unit_count(config, n_acc, n_iter),
    }
}

fn compute_cmma<R: Runtime>(
    client: &ComputeClient<R>,
    dtype: ElemType,
    config: LaunchConfig,
    n_acc: usize,
    n_iter: usize,
) -> KernelConfig {
    let client = client.clone();

    let kernel = Box::new(move || unsafe {
        let output_buffer = client.empty(1 * dtype.size());
        let output_buffer = BufferArg::Handle {
            handle: BufferBinding::from_raw_parts(output_buffer, 1),
        };

        compute_cmma_throughput::launch_unchecked(
            &client,
            CubeCount::Static(config.cube_count as u32, 1, 1),
            CubeDim::new_1d(config.cube_dim as u32),
            config.vector_size,
            output_buffer,
            n_acc,
            n_iter,
            dtype.into(),
        )
    });

    KernelConfig {
        kernel,
        unit_count: compute_unit_count(config, n_acc, n_iter),
    }
}

fn compute_unit_count(config: LaunchConfig, n_acc: usize, n_iter: usize) -> usize {
    2 * config.cube_count * config.cube_dim * n_iter * n_acc * config.vector_size
}

fn memory_direct<R: Runtime>(
    client: &ComputeClient<R>,
    dtype: ElemType,
    config: LaunchConfig,
    n_iter: usize,
) -> KernelConfig {
    let client = client.clone();

    let line_bytes = config.vector_size * dtype.size();

    const TARGET_BYTES: usize = 256 * 1024 * 1024;
    let max_alloc = client.properties().memory.max_page_size as usize;
    let target = TARGET_BYTES.min(max_alloc);

    let total_threads = config.cube_count * config.cube_dim;
    let num_lines = (target / line_bytes).max(total_threads);
    let bytes = num_lines * line_bytes;

    let kernel = Box::new(move || unsafe {
        let in_handle = client.empty(bytes);
        let out_handle = client.empty(bytes);

        memory_direct_throughput::launch_unchecked(
            &client,
            CubeCount::Static(config.cube_count as u32, 1, 1),
            CubeDim::new_1d(config.cube_dim as u32),
            config.vector_size,
            BufferArg::from_raw_parts(in_handle, num_lines),
            BufferArg::from_raw_parts(out_handle, num_lines),
            n_iter as usize,
            dtype.into(),
        )
    });

    KernelConfig {
        kernel,
        unit_count: 2 * num_lines * line_bytes * n_iter,
    }
}
