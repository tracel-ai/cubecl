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

#[cube]
pub fn swizzle(nth: u32, height: u32, #[comptime] swizzle_width: u32) -> (u32, u32) {
    let num_elem_per_swizzle_col = height * swizzle_width;

    let swizzle_id = nth % num_elem_per_swizzle_col;
    let swizzle_col = nth / num_elem_per_swizzle_col;

    let col_within_swizzle = swizzle_id / height;
    let col = swizzle_col * swizzle_width + col_within_swizzle;

    let topdown_row = swizzle_id % height;
    let is_bottom_up = swizzle_col % 2;

    let row = topdown_row + is_bottom_up * (height - 2 * topdown_row - 1);

    (row, col)
}
