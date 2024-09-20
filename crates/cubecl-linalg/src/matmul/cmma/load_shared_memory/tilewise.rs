use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma::{
    base::RuntimeCmmaInfo, block_io::base::BlockLoader, config::ComptimeCmmaInfo,
};

use super::{base::SmemLoader, load_info::LoadInfo, tiled_layout::TilingOrder};

pub(crate) struct TilewiseSmemLoader<I: LoadInfo, T: TilingOrder> {
    _load_info: PhantomData<I>,
    _tiling_order: PhantomData<T>,
}

#[cube]
impl<F: Float, FC: Float, I: LoadInfo, T: TilingOrder> SmemLoader<F, FC>
    for TilewiseSmemLoader<I, T>
{
    fn load_gmem_to_smem<L: BlockLoader<F, FC>>(
        gmem: &Tensor<F>,
        smem: &mut SharedMemory<FC>,
        k_offset: u32,
        runtime_info: RuntimeCmmaInfo,
        #[comptime] comptime_info: ComptimeCmmaInfo,
    ) {
        let tile_size = comptime_info.tile_size;
        let tensor_vec = vectorization_of(gmem);
        let ids = runtime_info.ids;

        let (skip_row, skip_col) = I::skips(k_offset, runtime_info);

        let coop_dim = comptime_info.coop_dim;
        let coop_id = ids.coop;

        let num_unit_reads = tile_size * tile_size / (tensor_vec * coop_dim);
        let num_units_per_row = tile_size / tensor_vec;

        let smem_tile_width = I::smem_width(comptime_info) / tile_size;
        let smem_tile_height = I::smem_height(comptime_info) / tile_size;
        let (tile_row, tile_col) = T::to_row_col(coop_id, smem_tile_width, smem_tile_height);

        let lane_row_step = coop_dim * tensor_vec / tile_size;
        let lane_row_offset = ids.lane / num_units_per_row;
        let read_row_offset = skip_row + tile_row * tile_size + lane_row_offset;

        let lane_col_offset = ids.lane % num_units_per_row * tensor_vec;
        let read_col = skip_col + tile_col * tile_size + lane_col_offset;

        let sm_stride = tile_size * tile_size;

        let write_offset = ids.coop * sm_stride + ids.lane * tensor_vec;
        let sm_step = coop_dim * tensor_vec;

        #[unroll]
        for i in 0..num_unit_reads {
            let read_row = read_row_offset + i * lane_row_step;
            let write_pos = write_offset + i * sm_step;

            L::load_single::<I>(gmem, smem, read_row, read_col, write_pos, runtime_info);
        }
    }

    fn get_tile_smem_position(
        tile_row: u32,
        tile_col: u32,
        #[comptime] comptime_info: ComptimeCmmaInfo,
    ) -> u32 {
        let tile_size = comptime_info.tile_size;
        let smem_stride = tile_size * tile_size;

        let smem_tile_width = I::smem_width(comptime_info) / tile_size;
        let smem_tile_height = I::smem_height(comptime_info) / tile_size;

        let nth_tile = T::to_nth_tile(tile_row, tile_col, smem_tile_width, smem_tile_height);
        nth_tile * smem_stride
    }
}
