use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma_matmul::BlockInfo;
use crate::matmul::tile_io::TileWriter;

#[derive(CubeType)]
// Writes the slice as they are given to it
pub struct DummySmemWriter<E: Numeric> {
    // TODO maybe shouldn't be owned, should have &'a
    pub memory: SharedMemory<Line<E>>,
    pub block_info: BlockInfo,
}

#[derive(CubeType)]
// Writes the slice as they are given to it
pub struct DummyTensorWriter<E: Numeric> {
    // TODO maybe shouldn't be owned, should have &'a
    pub memory: Tensor<Line<E>>,
    pub block_info: BlockInfo,
}

#[cube]
impl<E: Numeric> TileWriter<Line<E>> for DummySmemWriter<E> {
    fn write_with_cast<C: Numeric>(
        writer: &mut Self,
        slice: &Slice<'_, C>,
        compute_plane_offset: u32,
        accumulator_offset: u32,
    ) {
        let num_tile_elements = writer.block_info.tile_size_x * writer.block_info.tile_size_y;
        let num_tile_offset =
            compute_plane_offset * writer.block_info.num_tiles_y + accumulator_offset;

        let write_offset = num_tile_offset * num_tile_elements;
        for i in 0..num_tile_elements {
            writer.memory[i + write_offset] = Line::new(E::cast_from(slice[i]));
        }
    }
}

#[cube]
impl<E: Numeric> TileWriter<Line<E>> for DummyTensorWriter<E> {
    fn write_with_cast<C: Numeric>(
        writer: &mut Self,
        slice: &Slice<'_, C>,
        compute_plane_offset: u32,
        accumulator_offset: u32,
    ) {
    }
}

