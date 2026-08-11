use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_runtime::throughput::{KernelConfig, MemoryAccess, ThroughputKey};

use crate::throughput::{LaunchConfig, memory_probe::MemoryProbe};

/// Builds the copy kernel, moving `working_set` bytes per pass: half read out
/// of the input buffer, half written into the output one.
pub fn build_kernel<R: Runtime>(
    client: &ComputeClient<R>,
    key: ThroughputKey,
    config: LaunchConfig,
    working_set: usize,
) -> KernelConfig {
    let client = client.clone();
    let dtype = key.dtype();

    let line_bytes = config.vector_size * dtype.size();
    let probe = MemoryProbe::new(&client, config, line_bytes, MemoryAccess::Copy, working_set);
    let num_lines = probe.num_lines;

    let in_handle = client.empty(probe.buffer_bytes);
    let out_handle = client.empty(probe.buffer_bytes);

    let sample = Box::new(move |iterations: usize| {
        let start = cubecl_common::profile::Instant::now();
        unsafe {
            memory_direct_throughput::launch_unchecked(
                &client,
                CubeCount::Static(probe.cube_count as u32, 1, 1),
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

    // Each pass starts one line further into the buffer than the last. Same
    // lines, same count, same coalescing, but an address the compiler cannot
    // prove constant across passes: a working set small enough that every
    // thread copies a single line would otherwise have that copy sunk out of
    // the loop and performed once, and the probe would report a bandwidth the
    // hardware never moved. The rotation stays inside the buffers, so residency
    // is untouched — a working set that fits in cache still reports cache
    // bandwidth.
    let mut offset = 0;

    for _ in 0..n_iter {
        for step in 0..steps {
            let base = ABSOLUTE_POS + (step * stride);

            if base < len {
                let mut idx = base + offset;
                if idx >= len {
                    idx -= len;
                }

                output[idx] = input[idx];
            }
        }

        offset += 1;
        if offset == len {
            offset = 0;
        }
    }
}
