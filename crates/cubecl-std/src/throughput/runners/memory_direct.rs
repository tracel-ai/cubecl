use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_runtime::throughput::{KernelConfig, MemoryAccess, ThroughputKey};

use crate::throughput::{
    LaunchConfig,
    memory_probe::{self, MemoryProbe},
};

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

    // No real planes to coalesce across: give each unit a contiguous run
    // instead, which is what a blocked CPU kernel does.
    let blocked = config.plane_size == 1;

    let line_bytes = config.vector_size * dtype.size();
    let probe = MemoryProbe::new(
        &client,
        config,
        line_bytes,
        MemoryAccess::Copy,
        working_set,
        blocked,
    );

    let in_handle = client.empty(probe.buffer_bytes);
    memory_probe::prime(&client, &in_handle, probe.pool_lines, config, dtype);
    let out_handle = client.empty(probe.buffer_bytes);

    let sample = Box::new(move |iterations: usize| {
        let start = cubecl_common::profile::Instant::now();
        unsafe {
            memory_direct_throughput::launch_unchecked(
                &client,
                CubeCount::Static(probe.cube_count as u32, 1, 1),
                CubeDim::new(&client, config.cube_dim),
                config.vector_size,
                BufferArg::from_raw_parts(in_handle.clone(), probe.pool_lines),
                BufferArg::from_raw_parts(out_handle.clone(), probe.pool_lines),
                probe.window_lines,
                iterations,
                blocked,
                dtype,
            )
        };
        let _ = cubecl_core::future::block_on(client.sync());
        start.elapsed()
    });

    // One pass moves the window twice: once in, once out.
    let ops_count = 2 * probe.window_lines * config.vector_size;

    KernelConfig { sample, ops_count }
}

#[cube(launch_unchecked)]
pub fn memory_direct_throughput<I: Numeric, N: Size>(
    input: &[Vector<I, N>],
    output: &mut [Vector<I, N>],
    window: usize,
    n_iter: usize,
    #[comptime] blocked: bool,
    #[define(I)] _dtype: ElemType,
) {
    let len = output.len();
    let stride = CUBE_DIM as usize * CUBE_COUNT;

    // From `window` alone rather than from `window - ABSOLUTE_POS`, which
    // underflows for a thread past the end of a window smaller than the launch.
    // High threads get one step too many and the bounds check drops it.
    let steps = window.div_ceil(stride).max(1);

    // Each pass copies the *next* window of the buffers, not the same one
    // again. A window read repeatedly would be served from cache after the
    // first pass, and every working set below the cache would report cache
    // bandwidth instead of what a kernel of that size moves; coming back to a
    // window only after a whole buffer of traffic keeps it cold.
    //
    // It also keeps the addresses moving. A window small enough that every
    // thread copies a single line would otherwise be loop-invariant, and the
    // compiler is free to sink such a copy out of the loop and perform it once
    // — leaving the probe reporting a bandwidth the hardware never moved.
    let mut start = 0;
    let mut wrap = 0;

    for _ in 0..n_iter {
        for step in 0..steps {
            // Coalesced spreads one step's addresses across adjacent threads,
            // which is only fast where those threads share a real plane. A
            // CPU worker has no such neighbour, so it instead gets a run of
            // `steps` lines entirely its own.
            let base = if blocked {
                ABSOLUTE_POS * steps + step
            } else {
                ABSOLUTE_POS + (step * stride)
            };

            if base < window {
                let mut idx = start + base;
                if idx >= len {
                    idx -= len;
                }

                output[idx] = input[idx];
            }
        }

        start += window;
        // Back to the beginning, one line further along each time round, so a
        // window that fills the whole buffer still moves between passes.
        if start + window > len {
            wrap += 1;
            if wrap >= window {
                wrap = 0;
            }
            start = wrap;
        }
    }
}
