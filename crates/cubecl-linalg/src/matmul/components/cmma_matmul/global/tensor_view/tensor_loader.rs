use crate::matmul::components::cmma_matmul::global::ContinuousLoader;
use crate::matmul::components::global::{GmmConfig, Loader};
use crate::matmul::components::matrix::Ident;
use crate::matmul::components::stage::{LhsStageReader, RhsStageReader, Stage};
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
impl<EG: Numeric, ES: Numeric, G: GmmConfig> LhsTensorLoader<EG, ES, G> {
    pub fn new(
        tensor: Tensor<Line<EG>>,
        x_offset: u32,
        y_offset: u32,
        nth_batch: u32,
        #[comptime] config: G,
    ) -> Self {
        let stage = Stage::new::<G::SmmConfig>(Ident::Lhs, config.to_smm_config());
        let tensor_view = TensorView::new(tensor, x_offset, y_offset, nth_batch);

        LhsTensorLoader::<EG, ES, G> {
            tensor_view,
            stage,
            _e: PhantomData::<ES>.runtime(),
            _config: PhantomData::<G>.runtime(),
        }
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, G: GmmConfig> RhsTensorLoader<EG, ES, G> {
    pub fn new(
        tensor: Tensor<Line<EG>>,
        x_offset: u32,
        y_offset: u32,
        nth_batch: u32,
        #[comptime] config: G,
    ) -> Self {
        let stage = Stage::new::<G::SmmConfig>(Ident::Rhs, config.to_smm_config());
        let tensor_view = TensorView::new(tensor, x_offset, y_offset, nth_batch);

        RhsTensorLoader::<EG, ES, G> {
            tensor_view,
            stage,
            _e: PhantomData::<ES>.runtime(),
            _config: PhantomData::<G>.runtime(),
        }
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, G: GmmConfig> Loader<EG, ES, G> for LhsTensorLoader<EG, ES, G> {
    type StageReader = LhsStageReader<ES, G::SmmConfig>;

    fn fill_stage(this: &mut Self, #[comptime] config: G) -> Self::StageReader {
        ContinuousLoader::load_to_slice::<EG, ES, G>(
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
impl<EG: Numeric, ES: Numeric, G: GmmConfig> Loader<EG, ES, G> for RhsTensorLoader<EG, ES, G> {
    type StageReader = RhsStageReader<ES, G::SmmConfig>;

    fn fill_stage(this: &mut Self, #[comptime] config: G) -> Self::StageReader {
        ContinuousLoader::load_to_slice::<EG, ES, G>(
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
