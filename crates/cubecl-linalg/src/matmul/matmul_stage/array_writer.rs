use crate::matmul::matmul_global::{ArrayView, GlobalView};
use crate::matmul::matmul_stage::StageWriter;
use crate::matmul::stage_info::StageInfo;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
pub struct ArrayWriter<EG: Numeric> {
    pub array_view: ArrayView<EG>,
    pub stage_info: StageInfo,
}

#[cube]
impl<EG: Numeric> StageWriter<EG> for ArrayWriter<EG> {
    fn write<ES: Numeric>(
        stage_writer: &mut Self,
        slice: &Slice<'_, Line<ES>>,
        compute_plane_offset: u32,
        accumulator_offset: u32,
    ) {
        ArrayView::write_slice(
            &mut stage_writer.array_view,
            slice,
            compute_plane_offset * stage_writer.stage_info.tile_size_x,
            accumulator_offset * stage_writer.stage_info.tile_size_y,
            stage_writer.stage_info,
        );
    }
}
