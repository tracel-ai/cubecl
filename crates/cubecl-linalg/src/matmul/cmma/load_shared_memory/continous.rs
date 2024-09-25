use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};

use super::super::prologue::RuntimeCmmaInfo;
use crate::matmul::cmma::block_io::base::BlockLoader;
use crate::matmul::cmma::config::{ComptimeCmmaInfo, MainLoopStrategy};
use crate::matmul::cmma::load_shared_memory::base::get_tile_smem_index;

use super::base::SmemLoader;
use super::load_info::LoadInfo;
use super::tiled_layout::TilingOrder;

pub(crate) struct ContinuousSmemLoader {}

#[cube]
impl<F: Float, FC: Float, I: LoadInfo, T: TilingOrder> SmemLoader<F, FC, I, T>
    for ContinuousSmemLoader
{
    fn load_gmem_to_smem<L: BlockLoader<F, FC>>(
        gmem: &Tensor<F>,
        smem: &mut SharedMemory<FC>,
        k_offset: u32,
        runtime_info: RuntimeCmmaInfo,
        #[comptime] comptime_info: ComptimeCmmaInfo,
    ) {
        // Comptime information
        let plane_dim = comptime_info.plane_dim;
        let num_load_planes = match comptime_info.main_loop_strategy {
            MainLoopStrategy::Standard => comptime_info.num_compute_planes,
            MainLoopStrategy::Split(num_load_planes) => num_load_planes,
        };
        let num_smem_elements = I::smem_width(comptime_info) * I::smem_height(comptime_info);
        let vectorization = vectorization_of(gmem);
        let jump_length = num_load_planes * vectorization * plane_dim;
        let num_iterations = num_smem_elements / jump_length;
        let unroll = comptime_info.unroll;

        let lane_id = runtime_info.load_ids.lane;
        let plane_id = runtime_info.load_ids.plane;
        let unit_position_base = (plane_id * plane_dim + lane_id) * vectorization;
        let (skip_row, skip_col) = I::skips(k_offset, runtime_info);

        #[unroll(unroll)]
        for i in 0..num_iterations {
            let unit_position = unit_position_base + i * jump_length;
            let write_pos = unit_position;
            let (row, col) = apply_tiled_layout::<I, T>(unit_position, comptime_info);
            let read_row = row + skip_row;
            let read_col = col + skip_col;

            L::load_single::<I>(gmem, smem, read_row, read_col, write_pos, runtime_info)
        }
    }

    fn get_tile_smem_index(
        tile_row: u32,
        tile_col: u32,
        #[comptime] comptime_info: ComptimeCmmaInfo,
    ) -> u32 {
        get_tile_smem_index::<I, T>(tile_row, tile_col, comptime_info)
    }
}

#[cube]
pub(crate) fn apply_tiled_layout<I: LoadInfo, T: TilingOrder>(
    unit_position: u32,
    #[comptime] comptime_info: ComptimeCmmaInfo,
) -> (u32, u32) {
    let num_tile_elements = I::num_tile_elements(comptime_info);
    let smem_tile_width = I::smem_tile_width(comptime_info);
    let smem_tile_height = I::smem_tile_height(comptime_info);

    let nth_tile = unit_position / num_tile_elements;

    let (tile_row, tile_col) = T::to_row_col(nth_tile, smem_tile_width, smem_tile_height);

    let tile_stride = I::tile_width(comptime_info);
    let pos_within_tile = unit_position % num_tile_elements;
    let row_within_tile = pos_within_tile / tile_stride;
    let col_within_tile = pos_within_tile % tile_stride;

    let row = tile_row * I::tile_height(comptime_info) + row_within_tile;
    let col = tile_col * I::tile_width(comptime_info) + col_within_tile;

    (row, col)
}
