use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma_matmul::global::unload_from_slice;
use crate::matmul::matmul_global::{GmmConfig, Unloader};
use crate::matmul::matmul_stage::StageWriter;
use crate::matmul::matrix::Ident;

use super::TensorView;

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
pub fn new_tensor_unloader<EG: Numeric, G: GmmConfig>(
    tensor: Tensor<Line<EG>>,
    x_offset: u32,
    y_offset: u32,
) -> TensorUnloader<EG, G> {
    TensorUnloader::<EG, G> {
        tensor_view: TensorView::new(tensor, x_offset, y_offset),
        _config: PhantomData::<G>.runtime(),
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
        unload_from_slice::<EG, ES, G>(
            &mut this.tensor_view,
            slice,
            compute_plane_offset * config.stage_dim(Ident::Out).tile_size_x,
            accumulator_offset * config.stage_dim(Ident::Out).tile_size_y,
            config,
        )
    }
}
