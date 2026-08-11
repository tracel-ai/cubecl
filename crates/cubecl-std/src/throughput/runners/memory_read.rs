use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_runtime::throughput::{KernelConfig, MemoryAccess, ThroughputKey};

use crate::throughput::{LaunchConfig, memory_probe::MemoryProbe};

/// Builds the read-only streaming kernel, moving `working_set` bytes per pass,
/// all of them read.
///
/// This is [`memory_direct`](super::memory_direct) with the store removed. The
/// copy kernel moves a line in and a line back out, and counts both directions
/// in `ops_count`, so what it reports is total traffic across the memory
/// interface. That is the right ceiling for a kernel that also writes what it
/// reads, and the wrong one for a kernel that only reads — a weight stream, a
/// reduction, a gather. Those legitimately exceed the copy figure, because half
/// of the copy's traffic is a direction they never use.
///
/// Reported `ops_count` is the read count alone. Exactly one line is written,
/// by one thread, to keep the loads from being eliminated (see the kernel); at
/// hundreds of megabytes read that is not worth counting and is deliberately
/// left out of `ops_count` rather than approximated.
pub fn build_kernel<R: Runtime>(
    client: &ComputeClient<R>,
    key: ThroughputKey,
    config: LaunchConfig,
    working_set: usize,
) -> KernelConfig {
    let client = client.clone();
    let dtype = key.dtype();

    let line_bytes = config.vector_size * dtype.size();
    let probe = MemoryProbe::new(&client, config, line_bytes, MemoryAccess::Read, working_set);

    let in_handle = client.empty(probe.buffer_bytes);
    // One line: the kernel writes from a single thread, only to anchor the reads.
    let out_handle = client.empty(line_bytes);

    let sample = Box::new(move |iterations: usize| {
        let start = cubecl_common::profile::Instant::now();
        unsafe {
            memory_read_throughput::launch_unchecked(
                &client,
                CubeCount::Static(probe.cube_count as u32, 1, 1),
                CubeDim::new(&client, config.cube_dim),
                config.vector_size,
                BufferArg::from_raw_parts(in_handle.clone(), probe.pool_lines),
                BufferArg::from_raw_parts(out_handle.clone(), 1),
                probe.window_lines,
                iterations,
                dtype,
            )
        };
        let _ = cubecl_core::future::block_on(client.sync());
        start.elapsed()
    });

    // Reads only — no `2 *`. That factor is the whole difference from the copy.
    let ops_count = probe.window_lines * config.vector_size;

    KernelConfig { sample, ops_count }
}

#[cube(launch_unchecked)]
pub fn memory_read_throughput<I: Numeric, N: Size>(
    input: &[Vector<I, N>],
    output: &mut [Vector<I, N>],
    window: usize,
    n_iter: usize,
    #[define(I)] _dtype: ElemType,
) {
    let len = input.len();
    let stride = CUBE_DIM as usize * CUBE_COUNT;

    // From `window` alone rather than from `window - ABSOLUTE_POS`, which
    // underflows for a thread past the end of a window smaller than the launch.
    // High threads get one step too many and the bounds check drops it.
    let steps = window.div_ceil(stride).max(1);

    // Sum what is read. A load whose result is never used is dead code, and a
    // compiler that removes it turns this into a launch-overhead measurement
    // reporting an absurd bandwidth — so the reads have to reach an observable.
    let mut acc = Vector::<I, N>::empty();
    let lanes = acc.vector_size();
    #[unroll]
    for lane in 0..lanes {
        acc.insert(lane, I::cast_from(0));
    }

    // One accumulator, unlike `compute_direct`'s four. That kernel needs
    // independent chains because it is ALU-bound and would otherwise stall on
    // add latency; here the adds are free next to memory latency, and the
    // hiding comes from thread-level parallelism — `cube_count * cube_dim`
    // threads each with an independent address.
    //
    // Each pass reads the *next* window of the buffer, not the same one again.
    // A window read repeatedly would be served from cache after the first pass,
    // and every working set below the cache would report cache bandwidth
    // instead of what a kernel of that size moves; coming back to a window only
    // after a whole buffer of traffic keeps it cold.
    //
    // It also keeps the addresses moving. A window small enough that every
    // thread reads a single line would otherwise be loop-invariant, and the
    // compiler is free to hoist such a load out of the loop — leaving the probe
    // reporting the speed of adding a register to itself.
    let mut start = 0;
    let mut wrap = 0;

    for _ in 0..n_iter {
        for step in 0..steps {
            let base = ABSOLUTE_POS + (step * stride);

            if base < window {
                let mut idx = start + base;
                if idx >= len {
                    idx -= len;
                }

                acc += input[idx];
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

    // Guarded so the store cannot be hoisted out of the loop, and so the write
    // traffic is one line rather than one per thread. The compiler cannot prove
    // any given thread is not thread 0, so no thread's loads are dead.
    if ABSOLUTE_POS == 0 {
        output[0] = acc;
    }
}
