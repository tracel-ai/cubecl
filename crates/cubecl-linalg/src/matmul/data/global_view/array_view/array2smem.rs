use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::stage_info::StageInfo;

#[cube]
pub(crate) fn array_to_shared_memory<EG: Numeric, ES: Numeric>(
    gmem: &Array<Line<EG>>,
    smem: &mut SharedMemory<Line<ES>>,
    row_offset: u32,
    stage_info: StageInfo,
) {
    let stride_x = stage_info.num_tiles_y * stage_info.tile_size_y;

    let smem_tile_x =
        row_offset * stage_info.num_tiles_y * stage_info.tile_size_x * stage_info.tile_size_y;
    let gmem_tile_x = row_offset * stage_info.tile_size_x * stride_x;

    for tile_y in 0..stage_info.num_tiles_y {
        let smem_tile_y = tile_y * stage_info.tile_size_x * stage_info.tile_size_y;
        let gmem_tile_y = tile_y * stage_info.tile_size_y;

        for elem_x in 0..stage_info.tile_size_x {
            let smem_elem_x = elem_x * stage_info.tile_size_y;
            let gmem_elem_x = elem_x * stride_x;

            for elem_y in 0..stage_info.tile_size_y {
                let smem_offset = smem_tile_x + smem_tile_y + smem_elem_x + elem_y;
                let gmem_offset = gmem_tile_x + gmem_tile_y + gmem_elem_x + elem_y;

                smem[smem_offset] = Line::cast_from(gmem[gmem_offset]);
            }
        }
    }
}
