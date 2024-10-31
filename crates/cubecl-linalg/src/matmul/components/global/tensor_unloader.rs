use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::global::tilewise_unloading::TilewiseUnloading;
use crate::matmul::components::global::{GmmConfig, Unloader};
use crate::matmul::components::stage::StageWriter;

use super::tensor_view::TensorView;

#[derive(CubeType)]
pub struct TensorUnloader<EG: Numeric, G: GmmConfig> {
    pub tensor_view: TensorView<EG>,
    pub _config: PhantomData<G>,
}

#[cube]
impl<EG: Numeric, G: GmmConfig> Unloader<EG, G> for TensorUnloader<EG, G> {
    type StageWriter = Self;

    fn as_stage_writer(this: Self) -> Self::StageWriter {
        this
    }
}

#[cube]
impl<EG: Numeric, G: GmmConfig> TensorUnloader<EG, G> {
    pub fn new(tensor: Tensor<Line<EG>>, x_offset: u32, y_offset: u32, batch_offset: u32) -> Self {
        TensorUnloader::<EG, G> {
            tensor_view: TensorView::new(tensor, x_offset, y_offset, batch_offset),
            _config: PhantomData::<G>.runtime(),
        }
    }
}

#[cube]
impl<EG: Numeric, G: GmmConfig> StageWriter<EG, G> for TensorUnloader<EG, G> {
    fn write<ES: Numeric>(
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
