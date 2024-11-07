use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::global;
use crate::matmul::components::global::tensor_view::tilewise_unloading::TilewiseUnloading;
use crate::matmul::components::stage::StageWriter;

use super::base::TensorView;

#[derive(CubeType)]
pub struct Unloader<EG: Numeric> {
    pub tensor_view: TensorView<EG>,
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
    pub fn new(tensor: &Tensor<Line<EG>>, x_offset: u32, y_offset: u32, batch_offset: u32) -> Self {
        Unloader::<EG> {
            tensor_view: TensorView::new(tensor, x_offset, y_offset, batch_offset),
        }
    }
}

#[cube]
impl<EG: Numeric> StageWriter<EG> for Unloader<EG> {
    fn write<ES: Numeric, G: global::Config>(
        this: &mut Self,
        slice: &Slice<'_, Line<ES>>,
        compute_plane_offset: u32,
        accumulator_offset: u32,
        #[comptime] config: G,
    ) {
        TilewiseUnloading::unload_from_slice::<EG, ES, G>(
            &mut this.tensor_view,
            slice,
            compute_plane_offset,
            accumulator_offset,
            config,
        );
    }
}
