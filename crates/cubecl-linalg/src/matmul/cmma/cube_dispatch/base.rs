use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma::config::ComptimeCmmaInfo;

#[cube]
pub(crate) trait CubeDispatch {
    fn get_row_col(#[comptime] comptime_info: ComptimeCmmaInfo) -> (u32, u32);
}

pub(crate) struct RowMajorCubeDispatch {}

#[cube]
impl CubeDispatch for RowMajorCubeDispatch {
    fn get_row_col(#[comptime] comptime_info: ComptimeCmmaInfo) -> (u32, u32) {
        let block_size_m = comptime_info.block_size_m;
        let block_size_n = comptime_info.block_size_m;

        let cube_row = CUBE_POS_Y * block_size_m;
        let cube_col = CUBE_POS_X * block_size_n;

        (cube_row, cube_col)
    }
}

pub(crate) struct ColMajorCubeDispatch {}

#[cube]
impl CubeDispatch for ColMajorCubeDispatch {
    fn get_row_col(#[comptime] comptime_info: ComptimeCmmaInfo) -> (u32, u32) {
        let block_size_m = comptime_info.block_size_m;
        let block_size_n = comptime_info.block_size_m;

        let cube_row = CUBE_POS_X * block_size_m;
        let cube_col = CUBE_POS_Y * block_size_n;

        (cube_row, cube_col)
    }
}
