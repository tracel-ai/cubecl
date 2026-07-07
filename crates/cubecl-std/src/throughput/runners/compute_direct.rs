use cubecl::prelude::*;
use cubecl_core::{self as cubecl, frontend::fma, ir::ElemType};
use cubecl_runtime::throughput::{KernelConfig, LaunchConfig, ThroughputKey, ThroughputRunner};

const N_ITER: usize = 1024 * 8;

pub struct ComputeDirectRunner;

impl<R: Runtime> ThroughputRunner<R> for ComputeDirectRunner {
    fn build_kernel(
        client: &ComputeClient<R>,
        key: ThroughputKey,
        config: LaunchConfig,
    ) -> KernelConfig {
        let client = client.clone();
        let dtype = key.dtype;

        let has_fma = matches!(dtype, ElemType::Float(_));

        let kernel = Box::new(move || unsafe {
            let out = client.empty(config.vector_size * dtype.size());

            compute_direct_throughput::launch_unchecked(
                &client,
                CubeCount::Static(config.cube_count as u32, 1, 1),
                CubeDim::new_1d(config.cube_dim as u32),
                config.vector_size,
                BufferArg::from_raw_parts(out, 1),
                N_ITER,
                has_fma,
                dtype.into(),
            )
        });

        let ops_count = if has_fma { 8 } else { 4 }
            * config.cube_count
            * config.cube_dim
            * N_ITER
            * config.vector_size;

        KernelConfig { kernel, ops_count }
    }
}

#[cube(launch_unchecked)]
pub fn compute_direct_throughput<I: Numeric, N: Size>(
    output: &mut [Vector<I, N>],
    n_iter: usize,
    #[comptime] has_fma: bool,
    #[define(I)] _dtype: StorageType,
) {
    let tid = I::cast_from(ABSOLUTE_POS);

    let b = Vector::new(tid * I::cast_from(1) + I::cast_from(1));
    let c = Vector::new(tid * I::cast_from(1));

    let mut s0 = Vector::new(I::cast_from(1));
    let mut s1 = Vector::new(I::cast_from(1));
    let mut s2 = Vector::new(I::cast_from(1));
    let mut s3 = Vector::new(I::cast_from(1));

    for _ in 0..n_iter {
        if has_fma {
            s0 = fma(s0, b, c);
            s1 = fma(s1, b, c);
            s2 = fma(s2, b, c);
            s3 = fma(s3, b, c);
        } else {
            // gives lower bound as mul is slowest integer operation
            s0 = s0 * b;
            s1 = s1 * b;
            s2 = s2 * b;
            s3 = s3 * b;
        }
    }
    let sum = s0 + s1 + s2 + s3;

    if ABSOLUTE_POS == 0 {
        output[0] = sum;
    }
}
