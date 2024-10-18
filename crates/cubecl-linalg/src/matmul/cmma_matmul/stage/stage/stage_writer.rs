use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma_matmul::global::{write_slice, TensorView};
use crate::matmul::matmul_global::GmmConfig;
use crate::matmul::matmul_stage::StageWriter;
use crate::matmul::matrix::Ident;

#[derive(CubeType)]
pub struct OutStageWriter<EG: Numeric> {
    pub tensor_view: TensorView<EG>,
}

#[cube]
pub(crate) fn new_out_stage_writer<EG: Numeric, G: GmmConfig>(
    tensor_view: TensorView<EG>,
) -> OutStageWriter<EG> {
    OutStageWriter::<EG> { tensor_view }
}

#[cube]
impl<EG: Numeric> StageWriter<EG> for OutStageWriter<EG> {
    fn write<ES: Numeric, G: GmmConfig>(
        this: &mut Self,
        slice: &Slice<'_, Line<ES>>,
        compute_plane_offset: u32,
        accumulator_offset: u32,
        #[comptime] config: G,
    ) {
        write_slice::<EG, ES, G>(
            &mut this.tensor_view,
            slice,
            compute_plane_offset * config.stage_dim(Ident::Out).tile_size_x,
            accumulator_offset * config.stage_dim(Ident::Out).tile_size_y,
            config,
        )
    }
}
