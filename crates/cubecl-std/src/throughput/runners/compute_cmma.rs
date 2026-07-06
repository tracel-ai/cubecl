use cubecl::{ir::ElemType, prelude::*};
use cubecl_core as cubecl;
use cubecl_runtime::throughput::{KernelConfig, LaunchConfig, ThroughputRunner};

const N_ACC: usize = 8;
const N_ITER: usize = 1024;

pub struct ComputeCmmaRunner;

impl<R: Runtime> ThroughputRunner<R> for ComputeCmmaRunner {
    fn build_kernel(
        client: &ComputeClient<R>,
        dtype: ElemType,
        config: LaunchConfig,
    ) -> KernelConfig {
        let client = client.clone();

        let kernel = Box::new(move || unsafe {
            let out = client.empty(config.vector_size * dtype.size());

            compute_cmma_throughput::launch_unchecked(
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
pub fn compute_cmma_throughput<I: Numeric, N: Size>(
    output: &mut [Vector<I, N>],
    #[comptime] n_acc: usize,
    n_iter: usize,
    #[define(I)] _dtype: StorageType,
) {
    let m = 16 as usize;
    let n = 16 as usize;
    let k = 16 as usize;

    let a = cmma::Matrix::<I>::from_value(
        cmma::MatrixIdent::A,
        m,
        n,
        k,
        cmma::MatrixLayout::RowMajor,
        I::cast_from(1),
    );

    let b = cmma::Matrix::<I>::from_value(
        cmma::MatrixIdent::B,
        m,
        n,
        k,
        cmma::MatrixLayout::ColMajor,
        I::cast_from(1),
    );

    let mut acc = Sequence::<cmma::Matrix<I>>::new();

    #[unroll]
    for _ in 0..n_acc {
        acc.push(cmma::Matrix::<I>::from_value(
            cmma::MatrixIdent::Accumulator,
            m,
            n,
            k,
            cmma::MatrixLayout::Undefined,
            I::cast_from(0.0),
        ));
    }

    for _ in 0..n_iter {
        #[unroll]
        for i in 0..n_acc {
            cmma::execute(&a, &b, acc.index(i), acc.index(i));
        }
    }

    if ABSOLUTE_POS == 0 {
        cmma::store(output, acc.index(0), n as u32, cmma::MatrixLayout::RowMajor);
    }
}
