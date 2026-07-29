use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_runtime::throughput::{KernelConfig, ThroughputKey};

use crate::throughput::LaunchConfig;

/// Per-buffer size, clamped to the device's maximum allocation.
const TARGET_BYTES: usize = 512 * 1024 * 1024;

/// Builds the copy kernel.
pub fn build_kernel<R: Runtime>(
    client: &ComputeClient<R>,
    key: ThroughputKey,
    config: LaunchConfig,
) -> KernelConfig {
    let client = client.clone();
    let dtype = key.dtype();

    let line_bytes = config.vector_size * dtype.size();

    let max_alloc = client.properties().memory.max_page_size as usize;
    let target = TARGET_BYTES.min(max_alloc);

    let total_threads = config.cube_count * config.cube_dim;
    let num_lines = (target / line_bytes).max(total_threads);
    let bytes = num_lines * line_bytes;

    let in_handle = client.empty(bytes);
    let out_handle = client.empty(bytes);

    let sample = Box::new(move |iterations: usize| {
        let start = cubecl_common::profile::Instant::now();
        unsafe {
            memory_direct_throughput::launch_unchecked(
                &client,
                CubeCount::Static(config.cube_count as u32, 1, 1),
                CubeDim::new(&client, config.cube_dim),
                config.vector_size,
                BufferArg::from_raw_parts(in_handle.clone(), num_lines),
                BufferArg::from_raw_parts(out_handle.clone(), num_lines),
                iterations,
                dtype.into(),
            )
        };
        let _ = cubecl_core::future::block_on(client.sync());
        start.elapsed()
    });

    let ops_count = 2 * num_lines * config.vector_size;

    KernelConfig { sample, ops_count }
}

#[cube(launch_unchecked)]
pub fn memory_direct_throughput<I: Numeric, N: Size>(
    input: &[Vector<I, N>],
    output: &mut [Vector<I, N>],
    n_iter: usize,
    #[define(I)] _dtype: StorageType,
) {
    let len = output.len();
    let stride = CUBE_DIM as usize * CUBE_COUNT;

    let steps = (len - ABSOLUTE_POS).div_ceil(stride).max(1);

    for _ in 0..n_iter {
        for step in 0..steps {
            let idx = ABSOLUTE_POS + (step * stride);

            if idx < len {
                output[idx] = input[idx];
            }
        }
    }
}
