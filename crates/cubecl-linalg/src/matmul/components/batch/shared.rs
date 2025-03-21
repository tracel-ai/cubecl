use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::{
    Ident, MatmulPrecision,
    global::{self, GlobalConfig, IndexRange, IndexedQuantization, Quantization},
};
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};
use cubecl_std::{CubeOption, CubeOptionExpand, div_ceil};

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
    quantization: CubeOption<Quantization<MP::EG>>,
    #[comptime] config: GMM::Config,
) {
    let rank = out.rank();

    let batch_out = nth_batch * out.stride(rank - 2) * out.shape(rank - 2);
    let mut batch_lhs = 0u32.runtime();
    let mut batch_rhs = 0u32.runtime();
    for axis in 0..rank - 2 {
        let tmp = batch_out / out.stride(axis);
        batch_lhs += tmp % lhs.shape(axis) * lhs.stride(axis);
        batch_rhs += tmp % rhs.shape(axis) * rhs.stride(axis);
    }

    let indexed_quantization = match quantization {
        CubeOption::Some(quantization) => {
            // TODO Support broadcast
            //      The launcher should panic before executing the kernel
            //      if it is quantized and broadcasted.

            // Assuming that quantization params are stored in row major order per batch,
            // we compute the range of indices we need to retrieve the proper scaling for
            // the current global matmul.

            // LHS

            let num_stages_lhs_row_axis = div_ceil(
                lhs.shape(rank - 2),
                config.tiling_dimensions(Ident::Lhs).total_row(),
            );
            let num_stages_lhs_col_axis = div_ceil(
                lhs.shape(rank - 1),
                config.tiling_dimensions(Ident::Lhs).total_col(),
            );
            let num_stages_lhs_per_batch = num_stages_lhs_col_axis * num_stages_lhs_row_axis;

            let start_lhs =
                nth_batch * num_stages_lhs_per_batch + x_offset * num_stages_lhs_col_axis;

            let range_lhs = IndexRange {
                current: start_lhs + k_range.0,
                end: start_lhs + k_range.1,
                step: 1,
            };

            // RHS

            let num_stages_rhs_row_axis = div_ceil(
                rhs.shape(rank - 2),
                config.tiling_dimensions(Ident::Rhs).total_row(),
            );
            let num_stages_rhs_col_axis = div_ceil(
                rhs.shape(rank - 1),
                config.tiling_dimensions(Ident::Rhs).total_col(),
            );
            let num_stages_rhs_per_batch = num_stages_rhs_col_axis * num_stages_rhs_row_axis;

            let start_rhs = nth_batch * num_stages_rhs_per_batch + y_offset;

            let range_rhs = IndexRange {
                current: start_rhs + k_range.0 * num_stages_rhs_col_axis,
                end: start_rhs + k_range.1 * num_stages_rhs_col_axis,
                step: num_stages_rhs_col_axis,
            };

            // OUT
            let num_stages_out_row_axis = div_ceil(
                out.shape(rank - 2),
                config.tiling_dimensions(Ident::Out).total_row(),
            );
            let num_stages_out_col_axis = div_ceil(
                out.shape(rank - 1),
                config.tiling_dimensions(Ident::Out).total_col(),
            );
            let num_stages_out_per_batch = num_stages_out_col_axis * num_stages_out_row_axis;

            let index_out = num_stages_out_per_batch * nth_batch
                + x_offset * num_stages_out_col_axis
                + y_offset;

            CubeOption::new_Some(IndexedQuantization::new(
                quantization,
                range_lhs,
                range_rhs,
                index_out,
            ))
        }
        CubeOption::None => CubeOption::new_None(),
    };

    GMM::execute(
        GMM::init_lhs_loader(lhs, x_offset, k_range.0, nth_batch, batch_lhs, config),
        GMM::init_rhs_loader(rhs, k_range.0, y_offset, nth_batch, batch_rhs, config),
        GMM::init_unloader(out, x_offset, y_offset, nth_batch, batch_out),
        acc,
        k_range,
        indexed_quantization,
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
