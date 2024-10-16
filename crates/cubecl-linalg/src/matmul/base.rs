use cubecl_core::prelude::*;

use super::config::MatmulConfig;
use super::stage_info::StageInfos;

pub trait Matmul<I: Numeric, O: Numeric> {
    type Config: MatmulConfig;

    // TODO Can it migrate to config
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
