use cubecl_core::prelude::*;

use super::config::MatmulConfig;
use crate::matmul::config::MatmulLaunchConfig;

pub trait Matmul<I: Numeric, O: Numeric> {
    type Config: MatmulConfig;

    fn check_config(config: Self::Config);
}

pub trait MatmulLaunch<I: Numeric, O: Numeric> {
    type MatmulLaunchConfig: MatmulLaunchConfig;

    unsafe fn launch_unchecked<R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        lhs: TensorArg<'_, R>,
        rhs: TensorArg<'_, R>,
        out: TensorArg<'_, R>,
        config: Self::MatmulLaunchConfig,
    );
}
