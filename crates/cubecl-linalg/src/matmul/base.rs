use cubecl_core::prelude::*;

use super::config::MatmulConfig;

pub trait Matmul<I: Numeric, O: Numeric> {
    type Config: MatmulConfig;

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
