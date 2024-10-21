use crate::matmul::matmul_batch::BmmConfig;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::matmul_batch::BatchMatmul;

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
