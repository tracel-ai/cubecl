use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::{global, MatmulPrecision};
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};

#[cube]
/// Execute global matmul on lhs, rhs, writing in out.
/// x and y offsets are absolute rows and columns
pub(crate) fn gmm_execute<MP: MatmulPrecision, GMM: global::GlobalMatmul<MP>>(
    lhs: VirtualTensor<MP::EG>,
    rhs: VirtualTensor<MP::EG>,
    out: VirtualTensor<MP::EG, ReadWrite>,
    x_offset: u32,
    y_offset: u32,
    nth_batch: u32,
    acc: &mut GMM::Accumulator,
    k_range: (u32, u32),
    #[comptime] config: GMM::Config,
) {
    let rank = out.rank();
    let batch_out = nth_batch * out.shape(rank - 2) * out.shape(rank - 1);
    let mut batch_lhs = 0u32.runtime();
    let mut batch_rhs = 0u32.runtime();
    for b in 0..rank - 2 {
        let tmp = batch_out / out.stride(b);
        batch_lhs += tmp % lhs.shape(b) * lhs.stride(b);
        batch_rhs += tmp % rhs.shape(b) * rhs.stride(b);
    }

    GMM::execute(
        GMM::init_lhs_loader(lhs, x_offset, k_range.0, batch_lhs, config),
        GMM::init_rhs_loader(rhs, k_range.0, y_offset, batch_rhs, config),
        GMM::init_unloader(out, x_offset, y_offset, batch_out),
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
