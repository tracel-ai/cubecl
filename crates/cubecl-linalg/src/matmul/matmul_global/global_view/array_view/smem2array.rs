use crate::matmul::matmul_global::{ArrayView, GlobalView};
use crate::matmul::stage_info::StageInfo;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub(crate) fn smem_slice_to_gmem<EG: Numeric, ES: Numeric>(
    out: &mut ArrayView<EG>,
    slice: &Slice<'_, Line<ES>>,
    row_tile_begin: u32,
    col_tile_begin: u32,
    #[comptime] stage_info: StageInfo,
    #[comptime] slice_line_size: u32,
) {
    for elem_x in 0..stage_info.tile_size_x {
        let smem_elem_x = elem_x * stage_info.tile_size_y;

        for elem_y in 0..stage_info.tile_size_y {
            let smem_offset = smem_elem_x + elem_y;

            let write_row = row_tile_begin + elem_x;
            let write_col = col_tile_begin + elem_y;

            let value = slice[smem_offset / slice_line_size];
            ArrayView::write_coalesced::<ES>(out, write_row, write_col, value);
        }
    }
}
