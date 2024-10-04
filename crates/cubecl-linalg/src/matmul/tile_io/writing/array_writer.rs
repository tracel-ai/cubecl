use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma_matmul::BlockInfo;
use crate::matmul::tile_io::loading::array_into_row_major_block_layout;
use crate::matmul::tile_io::TileWriter;

#[derive(CubeType)]
pub struct ArrayWriter<E: Numeric> {
    pub gmem: Array<Line<E>>,
    pub block_info: BlockInfo,
}

#[cube]
impl<E: Numeric> TileWriter<Line<E>> for ArrayWriter<E> {
    fn write_with_cast<C: Numeric>(
        tile_writer: &mut Self,
        slice: &Slice<'_, C>,
        compute_plane_offset: u32,
        accumulator_offset: u32,
    ) {
        let num_tile_elements =
            tile_writer.block_info.tile_size_x * tile_writer.block_info.tile_size_y;
        let num_tile_offset =
            compute_plane_offset * tile_writer.block_info.num_tiles_y + accumulator_offset;

        let write_offset = num_tile_offset * num_tile_elements;

        // if UNIT_POS_Y == 1 && accumulator_offset == 0 {
        //     for i in 0..num_tile_elements {
        //         tile_writer.gmem[i + write_offset] = Line::cast_from(slice[i]);
        //     }
        // }
        array_into_row_major_block_layout(
            slice, // 256 elements
            tile_writer
                .gmem.as_slice_mut(),
                // .slice_mut(write_offset, write_offset + num_tile_elements), // can only write on 256 elements??
            tile_writer.block_info,
            true,
        );
    }
}
