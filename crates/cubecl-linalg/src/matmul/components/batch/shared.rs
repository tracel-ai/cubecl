use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::global;
use crate::matmul::components::global::{Loader, Unloader};

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
    acc: &mut GMM::Accumulator,
    k_range: (u32, u32),
    #[comptime] config: GMM::Config,
) {
    GMM::execute(
        GMM::Lhs::new::<GMM::Config>(lhs, x_offset, k_range.0, nth_batch, config),
        GMM::Rhs::new::<GMM::Config>(rhs, k_range.0, y_offset, nth_batch, config),
        GMM::Out::new(out, x_offset, y_offset, nth_batch),
        acc,
        k_range,
        config,
    );
}
