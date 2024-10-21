use crate::matmul::cmma_matmul::global::{load_to_slice, new_tensor_view, update_view};
use crate::matmul::cmma_matmul::stage::{
    as_slice_mut, new_lhs_stage_reader, new_rhs_stage_reader, new_stage,
};
use crate::matmul::cmma_matmul::stage::{LhsStageReader, RhsStageReader, Stage};
use crate::matmul::matmul_global::{GmmConfig, Loader};
use crate::matmul::matrix::Ident;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

use super::TensorView;

#[derive(CubeType)]
pub struct LhsTensorLoader<EG: Numeric, ES: Numeric, G: GmmConfig> {
    pub tensor_view: TensorView<EG>,
    pub stage: Stage<ES>,
    pub _e: PhantomData<ES>,
    pub _config: PhantomData<G>,
}

#[derive(CubeType)]
pub struct RhsTensorLoader<EG: Numeric, ES: Numeric, G: GmmConfig> {
    pub tensor_view: TensorView<EG>,
    pub stage: Stage<ES>,
    pub _e: PhantomData<ES>,
    pub _config: PhantomData<G>,
}

#[cube]
impl<EG: Numeric, ES: Numeric, G: GmmConfig> Loader<EG, ES, G> for LhsTensorLoader<EG, ES, G> {
    type StageReader = LhsStageReader<ES, G::SmmConfig>;

    fn fill_stage(this: &mut Self, #[comptime] config: G) -> Self::StageReader {
        load_to_slice::<EG, ES, G>(
            &this.tensor_view,
            as_slice_mut(&mut this.stage),
            Ident::Lhs,
            config,
        );
        new_lhs_stage_reader(this.stage)
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        update_view(&mut this.tensor_view, 0, k_offset, Ident::Lhs);
    }
}

#[cube]
pub fn new_lhs_tensor_loader<EG: Numeric, ES: Numeric, G: GmmConfig>(
    tensor: Tensor<Line<EG>>,
    x_offset: u32,
    y_offset: u32,
    #[comptime] config: G,
) -> LhsTensorLoader<EG, ES, G> {
    let stage = new_stage::<ES, G::SmmConfig>(Ident::Lhs, config.to_smm_config());
    let tensor_view = new_tensor_view(tensor, x_offset, y_offset);

    LhsTensorLoader::<EG, ES, G> {
        tensor_view,
        stage,
        _e: PhantomData::<ES>.runtime(),
        _config: PhantomData::<G>.runtime(),
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, G: GmmConfig> Loader<EG, ES, G> for RhsTensorLoader<EG, ES, G> {
    type StageReader = RhsStageReader<ES, G::SmmConfig>;

    fn fill_stage(this: &mut Self, #[comptime] config: G) -> Self::StageReader {
        load_to_slice::<EG, ES, G>(
            &this.tensor_view,
            as_slice_mut(&mut this.stage),
            Ident::Rhs,
            config,
        );
        new_rhs_stage_reader(this.stage)
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        update_view(&mut this.tensor_view, k_offset, 0, Ident::Rhs);
    }
}

#[cube]
pub fn new_rhs_tensor_loader<EG: Numeric, ES: Numeric, G: GmmConfig>(
    tensor: Tensor<Line<EG>>,
    x_offset: u32,
    y_offset: u32,
    #[comptime] config: G,
) -> RhsTensorLoader<EG, ES, G> {
    let stage = new_stage::<ES, G::SmmConfig>(Ident::Rhs, config.to_smm_config());
    let tensor_view = new_tensor_view(tensor, x_offset, y_offset);

    RhsTensorLoader::<EG, ES, G> {
        tensor_view,
        stage,
        _e: PhantomData::<ES>.runtime(),
        _config: PhantomData::<G>.runtime(),
    }
}
