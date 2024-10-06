use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma_matmul::BlockInfo;
use crate::matmul::tile_io::writing::smem2tensor::{Smem2Tensor, Smem2TensorSimple};
use crate::matmul::tile_io::TileWriter;

#[derive(CubeType)]
pub struct TensorWriter<E: Numeric> {
    pub gmem: Tensor<Line<E>>,
    pub cube_offsets: (u32, u32),
    pub block_info: BlockInfo,
}

#[cube]
pub(crate) fn new_tensor_writer<E: Numeric>(
    gmem: Tensor<Line<E>>,
    #[comptime] block_info: BlockInfo,
) -> TensorWriter<E> {
    TensorWriter::<E> {
        gmem,
        cube_offsets: (CUBE_POS_X, CUBE_POS_Y),
        block_info: block_info.runtime(),
    }
}

#[cube]
impl<E: Numeric> TileWriter<Line<E>> for TensorWriter<E> {
    fn write_with_cast<C: Numeric>(
        tile_writer: &mut Self,
        slice: &Slice<'_, C>,
        compute_plane_offset: u32,
        accumulator_offset: u32,
    ) {
        Smem2TensorSimple::smem_to_tensor(
            &mut tile_writer.gmem,
            slice,
            compute_plane_offset,
            accumulator_offset,
            tile_writer.cube_offsets,
            tile_writer.block_info,
        );
    }
}
