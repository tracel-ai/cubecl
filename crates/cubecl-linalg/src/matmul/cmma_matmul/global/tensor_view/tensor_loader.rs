use crate::matmul::cmma_matmul::global::load_to_slice;
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
            this.stage.as_slice_mut(),
            Ident::Lhs,
            config,
        );
        LhsStageReader::new(this.stage)
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, Ident::Lhs);
    }
}

#[cube]
pub fn new_lhs_tensor_loader<EG: Numeric, ES: Numeric, G: GmmConfig>(
    tensor: Tensor<Line<EG>>,
    x_offset: u32,
    y_offset: u32,
    #[comptime] config: G,
) -> LhsTensorLoader<EG, ES, G> {
    let stage = Stage::new::<G::SmmConfig>(Ident::Lhs, config.to_smm_config());
    let tensor_view = TensorView::new(tensor, x_offset, y_offset);

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
            this.stage.as_slice_mut(),
            Ident::Rhs,
            config,
        );
        RhsStageReader::new(this.stage)
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, Ident::Rhs);
    }
}

#[cube]
pub fn new_rhs_tensor_loader<EG: Numeric, ES: Numeric, G: GmmConfig>(
    tensor: Tensor<Line<EG>>,
    x_offset: u32,
    y_offset: u32,
    #[comptime] config: G,
) -> RhsTensorLoader<EG, ES, G> {
    let stage = Stage::new::<G::SmmConfig>(Ident::Rhs, config.to_smm_config());
    let tensor_view = TensorView::new(tensor, x_offset, y_offset);

    RhsTensorLoader::<EG, ES, G> {
        tensor_view,
        stage,
        _e: PhantomData::<ES>.runtime(),
        _config: PhantomData::<G>.runtime(),
    }
}
