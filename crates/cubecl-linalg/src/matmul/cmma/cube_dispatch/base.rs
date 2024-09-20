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
        let block_size_n = comptime_info.block_size_n;

        (CUBE_POS_Y * block_size_m, CUBE_POS_X * block_size_n)
    }
}

#[cube]
impl CubeDispatch for ColMajorCubeDispatch {
    fn get_row_col(#[comptime] comptime_info: ComptimeCmmaInfo) -> (u32, u32) {
        let block_size_m = comptime_info.block_size_m;
        let block_size_n = comptime_info.block_size_n;

        (CUBE_POS_X * block_size_m, CUBE_POS_Y * block_size_n)
    }
}

#[cube]
impl CubeDispatch for SwizzleCubeDispatch {
    #[allow(clippy::modulo_one)] // it somehow seems assumed that cube count is 1
    fn get_row_col(#[comptime] comptime_info: ComptimeCmmaInfo) -> (u32, u32) {
        let block_size_m = comptime_info.block_size_m;
        let block_size_n = comptime_info.block_size_n;

        let num_elem_per_swizzle_col = CUBE_COUNT_Y * 2;
        let nth_cube = CUBE_POS_X * CUBE_COUNT_Y + CUBE_POS_Y;
        let swizzle_id = nth_cube % num_elem_per_swizzle_col;

        let swizzle_col = nth_cube / num_elem_per_swizzle_col;
        let col_within_swizzle = swizzle_id / CUBE_COUNT_Y;
        let cube_col = swizzle_col * 2 + col_within_swizzle;

        let topdown_row = swizzle_id % CUBE_COUNT_Y;
        let is_bottom_up = (nth_cube / num_elem_per_swizzle_col) % 2;
        let cube_row = topdown_row + is_bottom_up * (CUBE_COUNT_Y - 2 * topdown_row - 1);

        (cube_row * block_size_m, cube_col * block_size_n)
    }
}
