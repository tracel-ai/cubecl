use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma_matmul::global::{unload_from_slice, TensorView};
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

