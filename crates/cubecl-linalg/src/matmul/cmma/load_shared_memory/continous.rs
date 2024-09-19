use std::marker::PhantomData;

use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};

use crate::matmul::cmma::block_io::base::BlockLoader;
use crate::matmul::cmma::load_shared_memory::tiled_layout::TiledPosition;
use crate::matmul::cmma::{base::RuntimeCmmaInfo, config::ComptimeCmmaInfo};

use super::base::SmemLoader;
use super::load_info::LoadInfo;
use super::tiled_layout::TilingOrder;

pub(crate) struct ContinuousSmemLoader<T: TilingOrder> {
    _tiling_order: PhantomData<T>,
}

#[cube]
impl<F: Float, FC: Float, T: TilingOrder> SmemLoader<F, FC> for ContinuousSmemLoader<T> {
    fn load_gmem_to_smem<I: LoadInfo, L: BlockLoader<F, FC>>(
        gmem: &Tensor<F>,
        smem: &mut SharedMemory<FC>,
        k_offset: u32,
        runtime_info: RuntimeCmmaInfo,
        #[comptime] comptime_info: ComptimeCmmaInfo,
    ) {
        // Comptime information
        let coop_dim = comptime_info.coop_dim;
        let num_coops = comptime_info.num_coops;
        let num_smem_elements = I::smem_width(comptime_info) * I::smem_height(comptime_info);
        let vectorization = vectorization_of(gmem);
        let jump_length = num_coops * vectorization * coop_dim;
        let num_iterations = num_smem_elements / jump_length;
        let unroll = comptime_info.unroll;

        let lane_id = runtime_info.ids.lane;
        let coop_id = runtime_info.ids.coop;
        let unit_position_base = lane_id * vectorization + coop_id * coop_dim;
        let (skip_row, skip_col) = I::skips(k_offset, runtime_info);

        #[unroll(unroll)]
        for i in 0..num_iterations {
            let unit_position = unit_position_base + i * jump_length;
            let write_pos = apply_smem_layout(unit_position);
            let tiled_position = apply_tiled_layout::<T>(
                unit_position,
                I::smem_width(comptime_info),
                I::smem_height(comptime_info),
                comptime_info,
            );
            let read_row = tiled_position.row * I::dim_horizontal(runtime_info) + skip_row;
            let read_col = tiled_position.col + skip_col;

            L::load_single::<I>(gmem, smem, read_row, read_col, write_pos, runtime_info)
        }
    }
}

#[cube]
fn apply_smem_layout(unit_position: u32) -> u32 {
    unit_position
}

#[cube]
pub(crate) fn apply_tiled_layout<T: TilingOrder>(
    absolute_position: u32,
    smem_width: u32,
    smem_height: u32,
    #[comptime] comptime_info: ComptimeCmmaInfo,
) -> TiledPosition {
    let tile_size = comptime_info.tile_size;
    let tile_square = tile_size * tile_size;
    let smem_tile_width = smem_width / tile_size;
    let smem_tile_height = smem_height / tile_size;

    let nth_tile = absolute_position / tile_square;

    let tile_row = T::tile_row(nth_tile, smem_tile_width, smem_tile_height);
    let tile_col = T::tile_col(nth_tile, smem_tile_width, smem_tile_height);

    let pos_within_tile = absolute_position % tile_square;
    let row_within_tile = pos_within_tile / tile_size;
    let col_within_tile = pos_within_tile % tile_size;

    let row = tile_row * tile_size + row_within_tile;
    let col = tile_col * tile_size + col_within_tile;

    TiledPosition { row, col }
}
