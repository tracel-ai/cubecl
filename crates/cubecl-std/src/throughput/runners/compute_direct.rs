use cubecl::{ir::ElemType, prelude::*};
use cubecl_core as cubecl;
use cubecl_runtime::throughput::{KernelConfig, LaunchConfig, ThroughputRunner};

const N_ACC: usize = 8;
const N_ITER: usize = 1024;

pub struct ComputeDirectRunner;

impl<R: Runtime> ThroughputRunner<R> for ComputeDirectRunner {
    fn build_kernel(
        client: &ComputeClient<R>,
        dtype: ElemType,
        config: LaunchConfig,
    ) -> KernelConfig {
        let client = client.clone();

        let kernel = Box::new(move || unsafe {
            let out = client.empty(config.vector_size * dtype.size());

            compute_direct_throughput::launch_unchecked(
                &client,
                CubeCount::Static(config.cube_count as u32, 1, 1),
                CubeDim::new_1d(config.cube_dim as u32),
                config.vector_size,
                BufferArg::from_raw_parts(out, 1),
                N_ACC,
                N_ITER,
                dtype.into(),
            )
        });

        let unit_count =
            2 * config.cube_count * config.cube_dim * N_ITER * N_ACC * config.vector_size;

        KernelConfig { kernel, unit_count }
    }
}

#[cube(launch_unchecked)]
pub fn compute_direct_throughput<I: Numeric, N: Size>(
    output: &mut [Vector<I, N>],
    #[comptime] n_acc: usize,
    n_iter: usize,
    #[define(I)] _dtype: StorageType,
) {
    let tid = I::cast_from(ABSOLUTE_POS);

    let b = Vector::new(tid * I::cast_from(1) + I::cast_from(1));
    let c = Vector::new(tid * I::cast_from(1));

    let mut acc = Array::new(n_acc);
    #[unroll]
    for i in 0..n_acc {
        acc[i] = Vector::new(tid + I::cast_from(i));
    }

    for _ in 0..n_iter {
        #[unroll]
        for i in 0..n_acc {
            acc[i] = acc[i] * b + c;
        }
    }

    let mut sum = Vector::zeroed();
    #[unroll]
    for i in 0..n_acc {
        sum += acc[i];
    }

    if ABSOLUTE_POS == 0 {
        output[0] = sum;
    }
}
