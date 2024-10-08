use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::block_info::BlockInfo;
use crate::matmul::tile_io::TileReader;

use super::tiled_layout::TilingOrder;

#[derive(CubeType)]
pub struct LhsSmemTileReader<E: Numeric, T: TilingOrder> {
    pub smem: SharedMemory<Line<E>>,
    pub block_info: BlockInfo,
    pub _tiling_order: PhantomData<T>,
}

#[derive(CubeType)]
pub struct RhsSmemTileReader<E: Numeric, T: TilingOrder> {
    pub smem: SharedMemory<Line<E>>,
    pub block_info: BlockInfo,
    pub _tiling_order: PhantomData<T>,
}

#[cube]
impl<E: Numeric, T: TilingOrder> TileReader<Line<E>> for LhsSmemTileReader<E, T> {
    fn read(
        reader: &Self,
        compute_plane_offset: u32,
        buffer_offset: u32,
        _accumulator_offset: u32,
    ) -> &Slice<'_, Line<E>> {
        let nth_tile = T::to_nth_tile(
            compute_plane_offset,
            buffer_offset,
            reader.block_info.num_tiles_y,
            reader.block_info.num_tiles_x,
        );

        let num_tile_elements = reader.block_info.tile_size_x * reader.block_info.tile_size_y;
        let start = nth_tile * num_tile_elements;
        reader.smem.slice(start, start + num_tile_elements)
    }
}

#[cube]
impl<E: Numeric, T: TilingOrder> TileReader<Line<E>> for RhsSmemTileReader<E, T> {
    fn read(
        reader: &Self,
        _compute_plane_offset: u32,
        buffer_offset: u32,
        accumulator_offset: u32,
    ) -> &Slice<'_, Line<E>> {
        let nth_tile = T::to_nth_tile(
            buffer_offset,
            accumulator_offset,
            reader.block_info.num_tiles_y,
            reader.block_info.num_tiles_x,
        );

        let num_tile_elements = reader.block_info.tile_size_x * reader.block_info.tile_size_y;
        let start = nth_tile * num_tile_elements;
        reader.smem.slice(start, start + num_tile_elements)
    }
}
