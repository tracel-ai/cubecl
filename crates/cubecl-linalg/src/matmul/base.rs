use cubecl_core::prelude::*;

use super::config::MatmulConfig;
use super::problem::MatmulProblem;
use super::stage_info::StageInfos;

pub trait Matmul<I: Numeric, O: Numeric> {
    type Config: MatmulConfig;

    fn can_process(problem: MatmulProblem) -> bool;

    // Can it migrate to config
    fn stage_infos() -> StageInfos;

    fn check_config(config: Self::Config);

    unsafe fn launch_unchecked<R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        lhs: TensorArg<'_, R>,
        rhs: TensorArg<'_, R>,
        out: TensorArg<'_, R>,
        config: Self::Config,
    );
}

// pub trait TensorMatmul<E: Numeric>: Matmul<E, E> {
//     unsafe fn launch_unchecked<R: Runtime>(
//         client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
//         cube_dim: CubeDim,
//         cube_count: CubeCount,
//         lhs: TensorArg<'_, R>,
//         rhs: TensorArg<'_, R>,
//         out: TensorArg<'_, R>,
//         layouts: (MatrixLayout, MatrixLayout),
//     );
// }

// pub trait FixedShapeMatmul<I: Numeric, O: Numeric>: Matmul<I, O> {
//     const M: u32;
//     const N: u32;
//     const K: u32;

//     unsafe fn launch_unchecked<R: Runtime>(
//         client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
//         cube_dim: CubeDim,
//         cube_count: CubeCount,
//         lhs: ArrayArg<'_, R>,
//         rhs: ArrayArg<'_, R>,
//         out: ArrayArg<'_, R>,
//         layouts: (MatrixLayout, MatrixLayout),
//     );
// }
