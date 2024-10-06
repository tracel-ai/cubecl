use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::block_info::BlockInfo;
use crate::matmul::tile_io::loading::smem_slice_to_gmem;
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
        smem_slice_to_gmem(
            slice,
            tile_writer.gmem.as_slice_mut(),
            compute_plane_offset,
            accumulator_offset,
            tile_writer.block_info,
        );
    }
}
