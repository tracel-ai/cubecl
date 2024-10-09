use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::matmul_stage::TileWriter;
use crate::matmul::stage_info::StageInfo;

#[derive(CubeType)]
pub struct ArrayWriter<E: Numeric> {
    pub gmem: Array<Line<E>>,
    pub block_info: StageInfo,
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

#[cube]
pub(crate) fn smem_slice_to_gmem<E: CubePrimitive, C: CubePrimitive>(
    smem_slice: &Slice<'_, E>,
    gmem: &mut SliceMut<'_, C>,
    row_offset: u32,
    col_offset: u32,
    block_info: StageInfo,
) {
    let stride_x = block_info.num_tiles_y * block_info.tile_size_y;

    let gmem_tile_x = row_offset * block_info.tile_size_x * stride_x;
    let gmem_tile_y = col_offset * block_info.tile_size_y;

    for elem_x in 0..block_info.tile_size_x {
        let smem_elem_x = elem_x * block_info.tile_size_y;
        let gmem_elem_x = elem_x * stride_x;

        for elem_y in 0..block_info.tile_size_y {
            let smem_offset = smem_elem_x + elem_y;
            let gmem_offset = gmem_tile_x + gmem_tile_y + gmem_elem_x + elem_y;

            gmem[gmem_offset] = C::cast_from(smem_slice[smem_offset]);
        }
    }
}
