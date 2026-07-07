use cubecl::prelude::*;
use cubecl_core as cubecl;
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

        let kernel = Box::new(move || unsafe {
            let out = client.empty(config.vector_size * dtype.size());

            compute_direct_throughput::launch_unchecked(
                &client,
                CubeCount::Static(config.cube_count as u32, 1, 1),
                CubeDim::new_1d(config.cube_dim as u32),
                config.vector_size,
                BufferArg::from_raw_parts(out, 1),
                N_ITER,
                dtype.into(),
            )
        });

        let unit_count = 2 * config.cube_count * config.cube_dim * N_ITER * config.vector_size;

        KernelConfig { kernel, unit_count }
    }
}

#[cube(launch_unchecked)]
pub fn compute_direct_throughput<I: Numeric, N: Size>(
    output: &mut [Vector<I, N>],
    n_iter: usize,
    #[define(I)] dtype: StorageType,
) {
    let tid = I::cast_from(ABSOLUTE_POS);

    let b = Vector::new(tid * I::cast_from(1) + I::cast_from(1));
    let c = Vector::new(tid * I::cast_from(1));

    let mut sum = Vector::zeroed();

    for _ in 0..n_iter {
        sum = sum * b + c;
    }

    if ABSOLUTE_POS == 0 {
        output[0] = sum;
    }
}
