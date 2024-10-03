use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma_matmul::BlockInfoR;
use crate::matmul::tile_io::TileReader;

#[derive(CubeType)]
pub struct SmemLhsReader<E: Numeric> {
    // TODO maybe shouldn't be owned, should have &'a
    pub memory: SharedMemory<Line<E>>,
    pub block_info: BlockInfoR,
}

#[derive(CubeType)]
pub struct SmemRhsReader<E: Numeric> {
    // TODO maybe shouldn't be owned, should have &'a
    pub memory: SharedMemory<Line<E>>,
    pub block_info: BlockInfoR,
}

#[cube]
impl<E: Numeric> TileReader<Line<E>> for SmemLhsReader<E> {
    fn read(
        reader: &Self,
        compute_plane_offset: u32,
        buffer_offset: u32,
        _accumulator_offset: u32,
    ) -> &Slice<'_, Line<E>> {
        let num_tile_elements = reader.block_info.tile_size_x * reader.block_info.tile_size_y;
        let num_tile_offset = compute_plane_offset * reader.block_info.num_tiles_y + buffer_offset;

        let start = num_tile_offset * num_tile_elements;
        reader.memory.slice(start, start + num_tile_elements)
    }
}

#[cube]
impl<E: Numeric> TileReader<Line<E>> for SmemRhsReader<E> {
    fn read(
        reader: &Self,
        _compute_plane_offset: u32,
        buffer_offset: u32,
        accumulator_offset: u32,
    ) -> &Slice<'_, Line<E>> {
        let num_tile_elements = reader.block_info.tile_size_x * reader.block_info.tile_size_y;
        let num_tile_offset = buffer_offset * reader.block_info.num_tiles_y + accumulator_offset;

        let start = num_tile_offset * num_tile_elements;
        reader.memory.slice(start, start + num_tile_elements)
    }
}
