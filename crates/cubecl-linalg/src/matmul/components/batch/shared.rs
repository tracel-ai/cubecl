use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::global;

#[cube]
/// Execute global matmul on lhs, rhs, writing in out.
/// x and y offsets are absolute rows and columns
pub(crate) fn gmm_execute<EG: Numeric, ES: Numeric, GMM: global::Matmul<EG, ES>>(
    lhs: &Tensor<Line<EG>>,
    rhs: &Tensor<Line<EG>>,
    out: &mut Tensor<Line<EG>>,
    x_offset: u32,
    y_offset: u32,
    nth_batch: u32,
    k_range: (u32, u32),
    #[comptime] config: GMM::Config,
) {
    GMM::execute(
        GMM::init_lhs_loader(lhs, x_offset, k_range.0, nth_batch, config),
        GMM::init_rhs_loader(rhs, k_range.0, y_offset, nth_batch, config),
        GMM::init_unloader(out, x_offset, y_offset, nth_batch),
        k_range,
        config,
    );
}
