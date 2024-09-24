use cubecl_core::cube;
use cubecl_core::{self as cubecl, prelude::*};

use super::super::config::ComptimeCmmaInfo;

use super::super::cube_dispatch::base::{
    ColMajorCubeDispatch, CubeDispatch, RowMajorCubeDispatch, SwizzleCubeDispatch,
};

#[derive(CubeType, Copy, Clone)]
/// Not divided by vectorization factor
///
/// Note: batch offsets take stride into account, but not the others
pub(crate) struct Offsets {
    pub batch_lhs: u32,
    pub batch_rhs: u32,
    pub batch_out: u32,
    pub cube_row: u32,
    pub cube_col: u32,
}

#[cube]
pub(crate) fn calculate_offsets<F: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    out: &Tensor<F>,
    #[comptime] comptime_info: ComptimeCmmaInfo,
) -> Offsets {
    let (cube_row, cube_col) = get_row_col(comptime_info);

    let rank = out.rank();

    let dim_m = lhs.shape(rank - 2);
    let dim_n = rhs.shape(rank - 1);

    // Batch offset for output
    let batch_out = dim_m * dim_n * CUBE_POS_Z;
    let mut batch_lhs = 0;
    let mut batch_rhs = 0;

    // Batch offset for lhs, rhs
    for b in 0..rank - 2 {
        let tmp = batch_out / out.stride(b);
        batch_lhs += tmp % lhs.shape(b) * lhs.stride(b);
        batch_rhs += tmp % rhs.shape(b) * rhs.stride(b);
    }

    Offsets {
        batch_lhs,
        batch_rhs,
        batch_out,
        cube_row,
        cube_col,
    }
}

#[cube]
pub(crate) fn get_row_col(#[comptime] comptime_info: ComptimeCmmaInfo) -> (u32, u32) {
    if comptime_info.cube_dispatch_strategy == 0 {
        RowMajorCubeDispatch::get_row_col(comptime_info)
    } else if comptime_info.cube_dispatch_strategy == 1 {
        ColMajorCubeDispatch::get_row_col(comptime_info)
    } else {
        SwizzleCubeDispatch::get_row_col(comptime_info)
    }
}
