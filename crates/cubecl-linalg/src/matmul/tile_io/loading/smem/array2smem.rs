use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma_matmul::BlockInfo;

// TODO
// WHEN READING, FROM 512 TO 256
// WHEN WRITING, FROM 256 TO 1024
// -> Separate the two algorithms

#[cube]
pub(crate) fn array_to_shared_memory<E: Numeric>(
    gmem: &Array<Line<E>>,
    smem: &mut SharedMemory<Line<E>>,
    block_info: BlockInfo,
) {
    let stride_x = block_info.num_tiles_y * block_info.tile_size_y;

    // TODO use plane id to not duplicate work
    for tile_x in 0..block_info.num_tiles_x {
        let tiled_offset_tile_x =
            tile_x * block_info.num_tiles_y * block_info.tile_size_x * block_info.tile_size_y;
        let continuous_offset_tile_x = tile_x * block_info.tile_size_x * stride_x;

        for tile_y in 0..block_info.num_tiles_y {
            let tiled_offset_tile_y = tile_y * block_info.tile_size_x * block_info.tile_size_y;
            let continuous_offset_tile_y = tile_y * block_info.tile_size_y;

            for elem_x in 0..block_info.tile_size_x {
                let tiled_offset_elem_x = elem_x * block_info.tile_size_y;
                let continuous_offset_elem_x = elem_x * stride_x;

                for elem_y in 0..block_info.tile_size_y {
                    let tiled_offset_elem_y = elem_y;
                    let continuous_offset_elem_y = elem_y;

                    let tiled_offset = tiled_offset_tile_x
                        + tiled_offset_tile_y
                        + tiled_offset_elem_x
                        + tiled_offset_elem_y;
                    let continuous_offset = continuous_offset_tile_x
                        + continuous_offset_tile_y
                        + continuous_offset_elem_x
                        + continuous_offset_elem_y;

                    smem[tiled_offset] = gmem[continuous_offset];
                }
            }
        }
    }
}

#[cube]
pub(crate) fn smem_slice_to_gmem<E: CubePrimitive, C: CubePrimitive>(
    original_slice: &Slice<'_, E>,
    slice_out: &mut SliceMut<'_, C>,
    block_info: BlockInfo,
) {
    let stride_x = block_info.num_tiles_y * block_info.tile_size_y;

    // TODO DO NOT ASSUME both slices are the same length
    for tile_x in 0..block_info.num_tiles_x {
        let tiled_offset_tile_x =
            tile_x * block_info.num_tiles_y * block_info.tile_size_x * block_info.tile_size_y;
        let continuous_offset_tile_x = tile_x * block_info.tile_size_x * stride_x;

        for tile_y in 0..block_info.num_tiles_y {
            let tiled_offset_tile_y = tile_y * block_info.tile_size_x * block_info.tile_size_y;
            let continuous_offset_tile_y = tile_y * block_info.tile_size_y;

            for elem_x in 0..block_info.tile_size_x {
                let tiled_offset_elem_x = elem_x * block_info.tile_size_y;
                let continuous_offset_elem_x = elem_x * stride_x;

                for elem_y in 0..block_info.tile_size_y {
                    let tiled_offset_elem_y = elem_y;
                    let continuous_offset_elem_y = elem_y;

                    let tiled_offset = tiled_offset_tile_x
                        + tiled_offset_tile_y
                        + tiled_offset_elem_x
                        + tiled_offset_elem_y;
                    let continuous_offset = continuous_offset_tile_x
                        + continuous_offset_tile_y
                        + continuous_offset_elem_x
                        + continuous_offset_elem_y;

                    slice_out[continuous_offset] = C::cast_from(original_slice[tiled_offset]);
                }
            }
        }
    }
}
