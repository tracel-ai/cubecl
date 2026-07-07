use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_runtime::throughput::{
    KernelConfig, LaunchConfig, MatrixSizes, ThroughputKey, ThroughputMode, ThroughputRunner,
};

const N_ITER: usize = 1024 * 8;

pub struct ComputeCmmaRunner;

impl<R: Runtime> ThroughputRunner<R> for ComputeCmmaRunner {
    fn build_kernel(
        client: &ComputeClient<R>,
        key: ThroughputKey,
        config: LaunchConfig,
    ) -> KernelConfig {
        let client = client.clone();
        let dtype = key.dtype;

        let matrix_sizes @ MatrixSizes { m, n, k } = match key.mode {
            ThroughputMode::ComputeCmma(matrix_sizes) => matrix_sizes,
            _ => unreachable!(),
        };

        let ops_per_cmma = 2 * m * n * k;

        let kernel = Box::new(move || unsafe {
            let out = client.empty(config.vector_size * dtype.size());

            compute_cmma_throughput::launch_unchecked(
                &client,
                CubeCount::Static(config.cube_count as u32, 1, 1),
                CubeDim::new_1d(config.cube_dim as u32),
                config.vector_size,
                BufferArg::from_raw_parts(out, 1),
                N_ITER,
                matrix_sizes,
                dtype.into(),
            )
        });

        let planes_per_cube = config.cube_dim / config.plane_size;
        let ops_count = config.cube_count * planes_per_cube * N_ITER * ops_per_cmma;

        KernelConfig { kernel, ops_count }
    }
}

#[cube(launch_unchecked)]
pub fn compute_cmma_throughput<I: Numeric, N: Size>(
    output: &mut [Vector<I, N>],
    n_iter: usize,
    #[comptime] matrix_sizes: MatrixSizes,
    #[define(I)] _dtype: StorageType,
) {
    let MatrixSizes { m, n, k } = matrix_sizes;

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

    let acc = cmma::Matrix::<I>::from_value(
        cmma::MatrixIdent::Accumulator,
        m,
        n,
        k,
        cmma::MatrixLayout::Undefined,
        I::cast_from(0.0),
    );

    for _ in 0..n_iter {
        cmma::execute(&a, &b, &acc, &acc);
    }

    if ABSOLUTE_POS == 0 {
        cmma::store(output, &acc, n as u32, cmma::MatrixLayout::RowMajor);
    }
}
