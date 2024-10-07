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
        let num_tile_elements = reader.block_info.tile_size_x * reader.block_info.tile_size_y;

        // TODO this assumes row major tiling order. Should match with tensor loader [to_nth_tile]
        // use T
        let num_tile_offset = compute_plane_offset * reader.block_info.num_tiles_y + buffer_offset;
        let start = num_tile_offset * num_tile_elements;

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
        let num_tile_elements = reader.block_info.tile_size_x * reader.block_info.tile_size_y;
        let num_tile_offset = buffer_offset * reader.block_info.num_tiles_y + accumulator_offset;
        // TODO use T

        let start = num_tile_offset * num_tile_elements;
        reader.smem.slice(start, start + num_tile_elements)
    }
}
