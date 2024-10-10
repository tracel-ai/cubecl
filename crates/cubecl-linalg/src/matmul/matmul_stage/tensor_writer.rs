use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::data::{GlobalView, TensorView};
use crate::matmul::matmul_stage::StageWriter;
use crate::matmul::stage_info::StageInfo;

#[derive(CubeType)]
pub struct TensorWriter<E: Numeric> {
    pub tensor_view: TensorView<E>,
    pub stage_info: StageInfo,
}

#[cube]
pub(crate) fn new_tensor_writer<E: Numeric>(
    tensor_view: TensorView<E>,
    #[comptime] block_info: StageInfo,
) -> TensorWriter<E> {
    TensorWriter::<E> {
        tensor_view,
        stage_info: block_info.runtime(),
    }
}

#[cube]
impl<E: Numeric> StageWriter<E> for TensorWriter<E> {
    fn write_with_cast<C: Numeric>(
        tile_writer: &mut Self,
        slice: &Slice<'_, C>,
        compute_plane_offset: u32,
        accumulator_offset: u32,
    ) {
        TensorView::write_slice(
            &mut tile_writer.tensor_view,
            slice,
            compute_plane_offset * tile_writer.stage_info.tile_size_x,
            accumulator_offset * tile_writer.stage_info.tile_size_y,
            tile_writer.stage_info,
        )
    }
}
