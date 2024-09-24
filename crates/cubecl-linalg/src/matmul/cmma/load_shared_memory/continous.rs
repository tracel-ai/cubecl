use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};

use super::super::runtime_info::RuntimeCmmaInfo;
use crate::matmul::cmma::block_io::base::BlockLoader;
use crate::matmul::cmma::config::ComptimeCmmaInfo;
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
        let coop_dim = comptime_info.coop_dim;
        let num_load_coops = comptime_info.num_load_coops;
        let num_smem_elements = I::smem_width(comptime_info) * I::smem_height(comptime_info);
        let vectorization = vectorization_of(gmem);
        let jump_length = num_load_coops * vectorization * coop_dim;
        let num_iterations = num_smem_elements / jump_length;
        let unroll = comptime_info.unroll;

        let lane_id = runtime_info.load_ids.lane;
        let coop_id = runtime_info.load_ids.coop;
        let unit_position_base = (coop_id * coop_dim + lane_id) * vectorization;
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
    let tile_size = comptime_info.tile_size;
    let tile_square = tile_size * tile_size;
    let smem_tile_width = I::smem_width(comptime_info) / tile_size;
    let smem_tile_height = I::smem_height(comptime_info) / tile_size;

    let nth_tile = unit_position / tile_square;

    let (tile_row, tile_col) = T::to_row_col(nth_tile, smem_tile_width, smem_tile_height);

    let pos_within_tile = unit_position % tile_square;
    let row_within_tile = pos_within_tile / tile_size;
    let col_within_tile = pos_within_tile % tile_size;

    let row = tile_row * tile_size + row_within_tile;
    let col = tile_col * tile_size + col_within_tile;

    (row, col)
}
