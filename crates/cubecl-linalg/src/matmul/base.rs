use super::config::MatmulConfig;
use super::matmul_batch::BatchMatmul;
use crate::matmul::matmul_batch::BmmConfig;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

pub trait Matmul<I: Numeric, O: Numeric> {
    type Config: MatmulConfig;

    fn check_config(config: Self::Config);
}

pub trait MatmulLaunch<I: Numeric, O: Numeric>: Matmul<I, O> {
    unsafe fn launch_unchecked<R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        lhs: TensorArg<'_, R>,
        rhs: TensorArg<'_, R>,
        out: TensorArg<'_, R>,
        config: <Self as Matmul<I, O>>::Config,
    );
}

#[cube(launch_unchecked)]
pub(crate) fn batch_matmul_launch<
    EG: Numeric,
    ES: Numeric,
    BMM: BatchMatmul<EG, B>,
    B: BmmConfig,
>(
    lhs: Tensor<Line<EG>>,
    rhs: Tensor<Line<EG>>,
    out: Tensor<Line<EG>>,
    #[comptime] config: B,
) {
    BMM::execute(lhs, rhs, out, config);
}
