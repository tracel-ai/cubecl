use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::global;
use crate::matmul::components::global::tensor_view::TensorWriter;
use crate::matmul::components::stage::StageWriter;

#[derive(CubeType)]
pub struct Unloader<EG: Numeric> {
    pub tensor_view: TensorWriter<EG>,
}

#[cube]
impl<EG: Numeric> global::Unloader<EG> for Unloader<EG> {
    type StageWriter = Self;

    fn as_stage_writer<G: global::Config>(this: Self) -> Self::StageWriter {
        this
    }
}

#[cube]
impl<EG: Numeric> Unloader<EG> {
    pub fn new(
        tensor: &mut Tensor<Line<EG>>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
    ) -> Self {
        // TMP
        Unloader::<EG> {
            tensor_view: TensorWriter::new(tensor, x_offset, y_offset, batch_offset),
        }
    }
}

#[cube]
impl<EG: Numeric> StageWriter<EG> for Unloader<EG> {
    fn write<ES: Numeric, G: global::Config>(
        this: &mut Self,
        slice: Slice<Line<ES>>,
        compute_plane_offset: u32,
        accumulator_offset: u32,
        #[comptime] config: G,
    ) {
    }
}
