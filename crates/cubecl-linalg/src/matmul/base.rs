use cubecl_core::prelude::*;

use super::matrix_layout::MatrixLayout;
use super::problem::{MatmulProblem, Requirements};
use super::stage_info::StageInfos;
use super::subroutine::Config;

pub trait Matmul<I: Numeric, O: Numeric> {
    fn can_process(problem: MatmulProblem) -> bool;
    fn requirements(problem: MatmulProblem) -> Requirements;

    fn stage_infos() -> StageInfos;
}

pub trait TensorMatmul<E: Numeric>: Matmul<E, E> {
    type Config: Config;

    unsafe fn launch_unchecked<R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        lhs: TensorArg<'_, R>,
        rhs: TensorArg<'_, R>,
        out: TensorArg<'_, R>,
        layouts: (MatrixLayout, MatrixLayout),
        config: Self::Config,
    );
}

pub trait FixedShapeMatmul<I: Numeric, O: Numeric>: Matmul<I, O> {
    const M: u32;
    const N: u32;
    const K: u32;
    type Config: Config;

    unsafe fn launch_unchecked<R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        lhs: ArrayArg<'_, R>,
        rhs: ArrayArg<'_, R>,
        out: ArrayArg<'_, R>,
        layouts: (MatrixLayout, MatrixLayout),
        config: Self::Config,
    );
}
