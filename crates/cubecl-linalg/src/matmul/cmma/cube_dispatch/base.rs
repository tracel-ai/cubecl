use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma::config::ComptimeCmmaInfo;

#[cube]
pub(crate) trait CubeDispatch {
    fn get_row_col(#[comptime] comptime_info: ComptimeCmmaInfo) -> (u32, u32);
}

pub(crate) struct RowMajorCubeDispatch {}
pub(crate) struct ColMajorCubeDispatch {}
pub(crate) struct SwizzleCubeDispatch {}

#[cube]
impl CubeDispatch for RowMajorCubeDispatch {
    fn get_row_col(#[comptime] comptime_info: ComptimeCmmaInfo) -> (u32, u32) {
        let block_size_m = comptime_info.block_size_m;
        let block_size_n = comptime_info.block_size_m;

        (CUBE_POS_Y * block_size_m, CUBE_POS_X * block_size_n)
    }
}

#[cube]
impl CubeDispatch for ColMajorCubeDispatch {
    fn get_row_col(#[comptime] comptime_info: ComptimeCmmaInfo) -> (u32, u32) {
        let block_size_m = comptime_info.block_size_m;
        let block_size_n = comptime_info.block_size_m;

        (CUBE_POS_X * block_size_m, CUBE_POS_Y * block_size_n)
    }
}

#[cube]
impl CubeDispatch for SwizzleCubeDispatch {
    fn get_row_col(#[comptime] comptime_info: ComptimeCmmaInfo) -> (u32, u32) {
        let block_size_m = comptime_info.block_size_m;
        let block_size_n = comptime_info.block_size_m;
        let swizzle_width = 2u32;
        let num_elem_per_swizzle_col = CUBE_COUNT * swizzle_width / CUBE_COUNT_X;

        let nth_cube = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
        let swizzle_id = nth_cube % num_elem_per_swizzle_col;

        let cube_col = swizzle_id % swizzle_width;
        let topdown_row = swizzle_id / swizzle_width;

        let is_bottom_up = (nth_cube / num_elem_per_swizzle_col) % 2;

        // if topdown, will equal topdown_row, else will equal CUBE_COUNT-topdown_row-1
        let cube_row = topdown_row + is_bottom_up * (CUBE_COUNT_Y - 2 * topdown_row - 1);

        (cube_row * block_size_m, cube_col * block_size_n)
    }
}
