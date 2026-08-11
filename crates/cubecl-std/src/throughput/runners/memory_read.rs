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
    let num_lines = probe.num_lines;

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
                BufferArg::from_raw_parts(in_handle.clone(), num_lines),
                BufferArg::from_raw_parts(out_handle.clone(), 1),
                iterations,
                dtype.into(),
            )
        };
        let _ = cubecl_core::future::block_on(client.sync());
        start.elapsed()
    });

    // Reads only — no `2 *`. That factor is the whole difference from the copy.
    let ops_count = num_lines * config.vector_size;

    KernelConfig { sample, ops_count }
}

#[cube(launch_unchecked)]
pub fn memory_read_throughput<I: Numeric, N: Size>(
    input: &[Vector<I, N>],
    output: &mut [Vector<I, N>],
    n_iter: usize,
    #[define(I)] _dtype: StorageType,
) {
    let len = input.len();
    let stride = CUBE_DIM as usize * CUBE_COUNT;

    let steps = (len - ABSOLUTE_POS).div_ceil(stride).max(1);

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
    // Each pass starts one line further into the buffer than the last. Same
    // lines, same count, same coalescing, but an address the compiler cannot
    // prove constant across passes: a working set small enough that every
    // thread reads a single line would otherwise have that load hoisted out of
    // the loop, and the probe would report the speed of adding a register to
    // itself. The rotation stays inside the buffer, so residency is untouched —
    // a working set that fits in cache still reports cache bandwidth.
    let mut offset = 0;

    for _ in 0..n_iter {
        for step in 0..steps {
            let base = ABSOLUTE_POS + (step * stride);

            if base < len {
                let mut idx = base + offset;
                if idx >= len {
                    idx -= len;
                }

                acc += input[idx];
            }
        }

        offset += 1;
        if offset == len {
            offset = 0;
        }
    }

    // Guarded so the store cannot be hoisted out of the loop, and so the write
    // traffic is one line rather than one per thread. The compiler cannot prove
    // any given thread is not thread 0, so no thread's loads are dead.
    if ABSOLUTE_POS == 0 {
        output[0] = acc;
    }
}
