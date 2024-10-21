use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::{Matmul, MatmulLaunch};

use super::BmmConfig;

#[cube]
pub trait BatchMatmul<EG: Numeric, B: BmmConfig>:
    'static + Send + Sync + Matmul<EG, EG, Config = B> + MatmulLaunch<EG, EG>
{
    fn execute(
        lhs: Tensor<Line<EG>>,
        rhs: Tensor<Line<EG>>,
        out: Tensor<Line<EG>>,
        #[comptime] config: Self::Config,
    );
}
