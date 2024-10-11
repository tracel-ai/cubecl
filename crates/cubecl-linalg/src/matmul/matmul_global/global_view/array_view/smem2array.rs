use crate::matmul::matmul_global::{ArrayView, GlobalView};
use crate::matmul::stage_info::StageInfo;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub(crate) fn smem_slice_to_gmem<E: Numeric, C: CubePrimitive>(
    out: &mut ArrayView<E>,
    smem_slice: &Slice<'_, C>,
    row_tile_begin: u32,
    col_tile_begin: u32,
    stage_info: StageInfo,
) {
    for elem_x in 0..stage_info.tile_size_x {
        let smem_elem_x = elem_x * stage_info.tile_size_y;

        for elem_y in 0..stage_info.tile_size_y {
            let smem_offset = smem_elem_x + elem_y;

            let write_row = row_tile_begin + elem_x;
            let write_col = col_tile_begin + elem_y;

            let value = smem_slice[smem_offset];
            ArrayView::write_coalesced::<C>(out, write_row, write_col, value);
        }
    }
}
