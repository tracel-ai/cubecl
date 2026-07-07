use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_runtime::throughput::{KernelConfig, LaunchConfig, ThroughputKey, ThroughputRunner};

pub struct MemoryDirectRunner;

impl<R: Runtime> ThroughputRunner<R> for MemoryDirectRunner {
    fn build_kernel(
        client: &ComputeClient<R>,
        key: ThroughputKey,
        config: LaunchConfig,
    ) -> KernelConfig {
        let client = client.clone();
        let dtype = key.dtype;

        let line_bytes = config.vector_size * dtype.size();

        const TARGET_BYTES: usize = 256 * 1024 * 1024;
        let max_alloc = client.properties().memory.max_page_size as usize;
        let target = TARGET_BYTES.min(max_alloc);

        let total_threads = config.cube_count * config.cube_dim;
        let num_lines = (target / line_bytes).max(total_threads);
        let bytes = num_lines * line_bytes;

        let kernel = Box::new(move |iterations: usize| unsafe {
            let in_handle = client.empty(bytes);
            let out_handle = client.empty(bytes);

            memory_direct_throughput::launch_unchecked(
                &client,
                CubeCount::Static(config.cube_count as u32, 1, 1),
                CubeDim::new_1d(config.cube_dim as u32),
                config.vector_size,
                BufferArg::from_raw_parts(in_handle, num_lines),
                BufferArg::from_raw_parts(out_handle, num_lines),
                iterations,
                dtype.into(),
            )
        });

        let ops_count = 2 * num_lines * config.vector_size;

        KernelConfig { kernel, ops_count }
    }
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
