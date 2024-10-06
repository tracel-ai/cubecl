use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma_matmul::BlockInfo;

#[cube]
pub(crate) fn array_to_shared_memory<E: Numeric>(
    gmem: &Array<Line<E>>,
    smem: &mut SharedMemory<Line<E>>,
    row_offset: u32,
    block_info: BlockInfo,
) {
    let stride_x = block_info.num_tiles_y * block_info.tile_size_y;

    let smem_tile_x =
        row_offset * block_info.num_tiles_y * block_info.tile_size_x * block_info.tile_size_y;
    let gmem_tile_x = row_offset * block_info.tile_size_x * stride_x;

    for tile_y in 0..block_info.num_tiles_y {
        let smem_tile_y = tile_y * block_info.tile_size_x * block_info.tile_size_y;
        let gmem_tile_y = tile_y * block_info.tile_size_y;

        for elem_x in 0..block_info.tile_size_x {
            let smem_elem_x = elem_x * block_info.tile_size_y;
            let gmem_elem_x = elem_x * stride_x;

            for elem_y in 0..block_info.tile_size_y {
                let smem_offset = smem_tile_x + smem_tile_y + smem_elem_x + elem_y;
                let gmem_offset = gmem_tile_x + gmem_tile_y + gmem_elem_x + elem_y;

                smem[smem_offset] = gmem[gmem_offset];
            }
        }
    }
}

#[cube]
pub(crate) fn smem_slice_to_gmem<E: CubePrimitive, C: CubePrimitive>(
    smem_slice: &Slice<'_, E>,
    gmem: &mut SliceMut<'_, C>,
    row_offset: u32,
    col_offset: u32,
    block_info: BlockInfo,
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
