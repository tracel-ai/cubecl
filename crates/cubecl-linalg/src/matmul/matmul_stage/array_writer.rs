use crate::matmul::matmul_global::{ArrayView, GlobalView};
use crate::matmul::matmul_stage::StageWriter;
use crate::matmul::stage_info::StageInfo;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
pub struct ArrayWriter<E: Numeric> {
    pub array_view: ArrayView<E>,
    pub stage_info: StageInfo,
}

#[cube]
impl<E: Numeric> StageWriter<E> for ArrayWriter<E> {
    fn write_with_cast<C: Numeric>(
        tile_writer: &mut Self,
        slice: &Slice<'_, C>,
        compute_plane_offset: u32,
        accumulator_offset: u32,
    ) {
        ArrayView::write_slice(
            &mut tile_writer.array_view,
            slice,
            compute_plane_offset * tile_writer.stage_info.tile_size_x,
            accumulator_offset * tile_writer.stage_info.tile_size_y,
            tile_writer.stage_info,
        );
    }
}
