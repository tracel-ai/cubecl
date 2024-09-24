use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma::config::ComptimeCmmaInfo;

#[cube]
pub(crate) trait Rasterization {
    fn get_row_col(#[comptime] comptime_info: ComptimeCmmaInfo) -> (u32, u32);
}

pub(crate) struct RowMajorRasterization {}
pub(crate) struct ColMajorRasterization {}
pub(crate) struct SwizzleRasterization {}

#[cube]
impl Rasterization for RowMajorRasterization {
    fn get_row_col(#[comptime] comptime_info: ComptimeCmmaInfo) -> (u32, u32) {
        let block_size_m = comptime_info.block_size_m;
        let block_size_n = comptime_info.block_size_n;

        (CUBE_POS_Y * block_size_m, CUBE_POS_X * block_size_n)
    }
}

#[cube]
impl Rasterization for ColMajorRasterization {
    fn get_row_col(#[comptime] comptime_info: ComptimeCmmaInfo) -> (u32, u32) {
        let block_size_m = comptime_info.block_size_m;
        let block_size_n = comptime_info.block_size_n;

        (CUBE_POS_X * block_size_m, CUBE_POS_Y * block_size_n)
    }
}

#[cube]
impl Rasterization for SwizzleRasterization {
    fn get_row_col(#[comptime] comptime_info: ComptimeCmmaInfo) -> (u32, u32) {
        let block_size_m = comptime_info.block_size_m;
        let block_size_n = comptime_info.block_size_n;

        let (cube_row, cube_col) = swizzle_two(CUBE_COUNT_Y, CUBE_POS_Y, CUBE_POS_X);
        (cube_row * block_size_m, cube_col * block_size_n)
    }
}

#[cube]
fn swizzle_two(height: u32, pos_vertical: u32, pos_horizontal: u32) -> (u32, u32) {
    let num_elem_per_swizzle_col = height * 2;
    let nth_cube = pos_horizontal * height + pos_vertical;
    let swizzle_id = nth_cube % num_elem_per_swizzle_col;

    let swizzle_col = nth_cube / num_elem_per_swizzle_col;
    let col_within_swizzle = swizzle_id / height;
    let cube_col = swizzle_col * 2 + col_within_swizzle;

    let topdown_row = swizzle_id % height;
    let is_bottom_up = (nth_cube / num_elem_per_swizzle_col) % 2;
    let cube_row = topdown_row + is_bottom_up * (height - 2 * topdown_row - 1);

    (cube_row, cube_col)
}
