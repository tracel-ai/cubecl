use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma_matmul::config::CmmaConfig;
use crate::matmul::matmul_global::{TensorView, WriteView};
use crate::matmul::matmul_stage::StageWriter;

#[derive(CubeType)]
pub struct TensorWriter<EG: Numeric> {
    pub tensor_view: TensorView<EG>,
}

#[cube]
pub(crate) fn new_tensor_writer<EG: Numeric>(tensor_view: TensorView<EG>) -> TensorWriter<EG> {
    TensorWriter::<EG> { tensor_view }
}

#[cube]
impl<EG: Numeric> StageWriter<EG> for TensorWriter<EG> {
    type Config = CmmaConfig;

    fn write<ES: Numeric>(
        stage_writer: &mut Self,
        slice: &Slice<'_, Line<ES>>,
        compute_plane_offset: u32,
        accumulator_offset: u32,
        #[comptime] slice_line_size: u32,
        #[comptime] config: Self::Config,
    ) {
        TensorView::write_slice(
            &mut stage_writer.tensor_view,
            slice,
            compute_plane_offset * config.stage_dims.out.tile_size_x,
            accumulator_offset * config.stage_dims.out.tile_size_y,
            slice_line_size,
            config,
        )
    }
}
