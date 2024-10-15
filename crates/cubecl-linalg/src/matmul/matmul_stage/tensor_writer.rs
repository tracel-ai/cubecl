use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::matmul_global::{GlobalView, TensorView};
use crate::matmul::matmul_stage::StageWriter;
use crate::matmul::stage_info::StageInfo;

#[derive(CubeType)]
pub struct TensorWriter<EG: Numeric> {
    pub tensor_view: TensorView<EG>,
    pub stage_info: StageInfo,
}

#[cube]
pub(crate) fn new_tensor_writer<EG: Numeric>(
    tensor_view: TensorView<EG>,
    #[comptime] block_info: StageInfo,
) -> TensorWriter<EG> {
    TensorWriter::<EG> {
        tensor_view,
        stage_info: block_info.runtime(),
    }
}

#[cube]
impl<EG: Numeric> StageWriter<EG> for TensorWriter<EG> {
    fn write<ES: Numeric>(
        stage_writer: &mut Self,
        slice: &Slice<'_, Line<ES>>,
        compute_plane_offset: u32,
        accumulator_offset: u32,
        #[comptime] slice_line_size: u32,
    ) {
        TensorView::write_slice(
            &mut stage_writer.tensor_view,
            slice,
            compute_plane_offset * stage_writer.stage_info.tile_size_x,
            accumulator_offset * stage_writer.stage_info.tile_size_y,
            stage_writer.stage_info,
            slice_line_size,
        )
    }
}
