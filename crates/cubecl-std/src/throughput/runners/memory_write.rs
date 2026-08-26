use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_runtime::throughput::{KernelConfig, MemorySpec, ThroughputKey};

use crate::throughput::{LaunchConfig, memory_probe::MemoryProbe};

/// Builds the write-only streaming kernel, moving `working_set` bytes per
/// pass, all of them written.
///
/// This is [`memory_direct`](super::memory_direct) with the load removed. The
/// copy kernel moves a line in and a line back out, and counts both
/// directions in `ops_count`, so what it reports is total traffic across the
/// memory interface. That is the right ceiling for a kernel that also reads
/// what it writes, and the wrong one for a kernel that only writes: an RNG
/// fill, a memset, a broadcast. Those legitimately exceed the copy figure,
/// because half of the copy's traffic is a direction they never use.
///
/// Reported `ops_count` is the write count alone.
pub fn build_kernel<R: Runtime>(
    client: &ComputeClient<R>,
    key: ThroughputKey,
    config: LaunchConfig,
    spec: MemorySpec,
) -> KernelConfig {
    let client = client.clone();
    let dtype = key.dtype();

    let line_bytes = config.vector_size * dtype.size();
    let probe = MemoryProbe::new(&client, config, line_bytes, spec);

    let out_handle = client.empty(probe.buffer_bytes);

    let sample = Box::new(move |iterations: usize| {
        let start = cubecl_common::profile::Instant::now();
        unsafe {
            memory_write_throughput::launch_unchecked(
                &client,
                CubeCount::Static(probe.cube_count as u32, 1, 1),
                CubeDim::new(&client, config.cube_dim),
                config.vector_size,
                BufferArg::from_raw_parts(out_handle.clone(), probe.pool_lines),
                probe.window_lines,
                iterations,
                probe.blocked,
                probe.walks(),
                dtype,
            )
        };
        let _ = cubecl_core::future::block_on(client.sync());
        start.elapsed()
    });

    // Writes only, no `2 *`. That factor is the whole difference from the copy.
    let ops_count = probe.window_lines * config.vector_size;

    KernelConfig {
        sample,
        ops_count,
        min_iterations: probe.min_iterations(),
    }
}

#[cube(launch_unchecked)]
pub fn memory_write_throughput<I: Numeric, N: Size>(
    output: &mut [Vector<I, N>],
    window: usize,
    n_iter: usize,
    #[comptime] blocked: bool,
    #[comptime] walks: bool,
    #[define(I)] _dtype: ElemType,
) {
    let len = output.len();
    let stride = CUBE_DIM as usize * CUBE_COUNT;

    // From `window` alone rather than from `window - ABSOLUTE_POS`, which
    // underflows for a thread past the end of a window smaller than the launch.
    // High threads get one step too many and the bounds check drops it.
    let steps = window.div_ceil(stride).max(1);

    // Read once, write everywhere: the reverse of `memory_read`, which reads
    // everywhere and writes once. `n_iter` is a scalar the kernel compiler
    // sees only at launch time, so it cannot fold the stores into a single
    // known-pattern fill, and the per-lane offset keeps the written line from
    // being a uniform value even within one call.
    let seed = I::cast_from(n_iter);
    let mut line = Vector::<I, N>::empty();
    let lanes = line.vector_size();
    #[unroll]
    for lane in 0..lanes {
        line.insert(lane, seed + I::cast_from(lane));
    }

    // Each pass writes the *next* window of the buffer, not the same one
    // again. A window written repeatedly would stay resident in cache after
    // the first pass, and every working set below the cache would report
    // cache bandwidth instead of what a kernel of that size moves; coming
    // back to a window only after a whole buffer of traffic keeps it cold.
    //
    // It also keeps the addresses moving. A window small enough that every
    // thread writes a single line would otherwise be loop-invariant, and the
    // compiler is free to sink such a store out of the loop and perform it
    // once, leaving the probe reporting a bandwidth the hardware never moved.
    let mut start = 0;
    let mut wrap = 0;
    let mut turn = 0;
    let mut mark = 0;

    for _ in 0..n_iter {
        for step in 0..steps {
            // A window with no pool to walk turns inside each worker's own run
            // instead. The addresses still move between passes, without which
            // the compiler folds a pass that rewrites its own bytes away and
            // the probe reports twice what the bus can carry, but a line never
            // changes hands and so is never pulled from the cache holding it.
            let place = if walks {
                step
            } else {
                let mut turned = step + turn;
                if turned >= steps {
                    turned -= steps;
                }
                turned
            };

            // Coalesced spreads one step's addresses across adjacent threads,
            // which is only fast where those threads share a real plane. A
            // CPU worker has no such neighbour, so it instead gets a run of
            // `steps` lines entirely its own.
            let base = if blocked {
                ABSOLUTE_POS * steps + place
            } else {
                ABSOLUTE_POS + (place * stride)
            };

            if base < window {
                let mut idx = start + base;
                if idx >= len {
                    idx -= len;
                }

                output[idx] = line;
            }
        }

        if walks {
            start += window;
            // Back to the beginning, one line further along each time round,
            // so a window that fills the whole buffer still moves between
            // passes. The test is whether the window starts past the end, not
            // whether it reaches past: the index wraps, so the last position
            // of a cycle straddles the end rather than being skipped.
            if start >= len {
                wrap += 1;
                if wrap >= window {
                    wrap = 0;
                }
                start = wrap;
            }
        } else {
            turn += 1;
            if turn >= steps {
                turn = 0;
            }

            // A pass with no pool to walk writes the bytes the pass before it
            // wrote, and the compiler is entitled to keep only the last of
            // them. It cannot once no two passes store the same value, which
            // costs one line of arithmetic against a window of stores.
            mark += 1;
            #[unroll]
            for lane in 0..lanes {
                line.insert(lane, seed + I::cast_from(mark) + I::cast_from(lane));
            }
        }
    }
}
