use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma_matmul::{into_runtime, BlockInfo, BlockInfoR};
use crate::matmul::tile_io::TileWriter;

#[derive(CubeType)]
pub struct GmemTensorWriter<E: Numeric> {
    pub gmem: Tensor<Line<E>>,
    pub cube_offsets: (u32, u32),
    pub block_info: BlockInfoR,
}

#[cube]
pub(crate) fn new_tensor_writer<E: Numeric>(
    gmem: Tensor<Line<E>>,
    #[comptime] block_info: BlockInfo,
) -> GmemTensorWriter<E> {
    GmemTensorWriter::<E> {
        gmem,
        cube_offsets: (CUBE_POS_X, CUBE_POS_Y),
        block_info: into_runtime(block_info),
    }
}

#[cube]
impl<'a, E: Numeric> TileWriter<Line<E>> for GmemTensorWriter<E> {
    type Gmem = Tensor<Line<E>>;

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
        for i in 0..num_tile_elements {
            tile_writer.gmem[i + write_offset] = Line::new(E::cast_from(slice[i]));
        }
    }
}
