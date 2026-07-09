use cubecl::prelude::*;
use cubecl_core::{self as cubecl, frontend::fma, ir::ElemType};
use cubecl_runtime::throughput::{KernelConfig, ThroughputKey};

use crate::throughput::LaunchConfig;

pub fn build_kernel<R: Runtime>(
    client: &ComputeClient<R>,
    key: ThroughputKey,
    config: LaunchConfig,
) -> KernelConfig {
    let client = client.clone();
    let dtype = key.dtype;

    let use_fma = matches!(dtype, ElemType::Float(_));

    let kernel = Box::new(move |iterations: usize| unsafe {
        let out = client.empty(config.vector_size * dtype.size());

        compute_direct_throughput::launch_unchecked(
            &client,
            CubeCount::Static(config.cube_count as u32, 1, 1),
            CubeDim::new(&client, config.cube_dim),
            config.vector_size,
            BufferArg::from_raw_parts(out, 1),
            iterations,
            use_fma,
            dtype.into(),
        )
    });

    // `CHAINS` independent accumulators per lane, each retiring one fma (two flops) or one mul.
    let ops_per_chain = if use_fma { 2 } else { 1 };
    let ops_count =
        ops_per_chain * CHAINS * config.cube_count * config.cube_dim * config.vector_size;

    KernelConfig { kernel, ops_count }
}

/// Independent accumulator chains per lane, to hide the arithmetic latency of the dependent
/// `sum = fma(sum, b, c)` recurrence.
const CHAINS: usize = 4;

#[cube(launch_unchecked)]
pub fn compute_direct_throughput<I: Numeric, N: Size>(
    output: &mut [Vector<I, N>],
    n_iter: usize,
    #[comptime] use_fma: bool,
    #[define(I)] _dtype: StorageType,
) {
    let tid = I::cast_from(ABSOLUTE_POS);

    let mut b = Vector::<I, N>::empty();
    let mut c = Vector::<I, N>::empty();

    let mut s0 = Vector::<I, N>::empty();
    let mut s1 = Vector::<I, N>::empty();
    let mut s2 = Vector::<I, N>::empty();
    let mut s3 = Vector::<I, N>::empty();

    // Every lane and every chain gets a distinct seed. Values that are provably equal get folded:
    // a broadcast (`Vector::new`) collapses the vector to one lane, and identically seeded chains
    // collapse to one chain. Either way the kernel retires fewer operations than `ops_count` bills.
    let lanes = b.vector_size();
    #[unroll]
    for lane in 0..lanes {
        let offset = I::cast_from(lane);
        b.insert(lane, tid + offset + I::cast_from(1));
        c.insert(lane, tid + offset);

        s0.insert(lane, offset + I::cast_from(1));
        s1.insert(lane, offset + I::cast_from(2));
        s2.insert(lane, offset + I::cast_from(3));
        s3.insert(lane, offset + I::cast_from(4));
    }

    for _ in 0..n_iter {
        if use_fma {
            s0 = fma(s0, b, c);
            s1 = fma(s1, b, c);
            s2 = fma(s2, b, c);
            s3 = fma(s3, b, c);
        } else {
            // gives lower bound as mul is slowest integer operation
            s0 *= b;
            s1 *= b;
            s2 *= b;
            s3 *= b;
        }
    }

    let sum = s0 + s1 + s2 + s3;

    if ABSOLUTE_POS == 0 {
        output[0] = sum;
    }
}
