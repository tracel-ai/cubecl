use crate::matmul::components::global::continuous_loading::ContinuousLoading;
use crate::matmul::components::global::{Config, Loader};
use crate::matmul::components::matrix::Ident;
use crate::matmul::components::stage::{LhsReader, RhsReader, Stage};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

use super::tensor_view::TensorView;

#[derive(CubeType)]
pub struct LhsTensorLoader<EG: Numeric, ES: Numeric, G: Config> {
    pub tensor_view: TensorView<EG>,
    pub stage: Stage<ES>,
    pub _e: PhantomData<ES>,
    pub _config: PhantomData<G>,
}

#[derive(CubeType)]
pub struct RhsTensorLoader<EG: Numeric, ES: Numeric, G: Config> {
    pub tensor_view: TensorView<EG>,
    pub stage: Stage<ES>,
    pub _e: PhantomData<ES>,
    pub _config: PhantomData<G>,
}

#[cube]
impl<EG: Numeric, ES: Numeric, G: Config> LhsTensorLoader<EG, ES, G> {
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
impl<EG: Numeric, ES: Numeric, G: Config> RhsTensorLoader<EG, ES, G> {
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
impl<EG: Numeric, ES: Numeric, G: Config> Loader<EG, ES, G> for LhsTensorLoader<EG, ES, G> {
    type StageReader = LhsReader<ES, G::SmmConfig>;

    fn fill_stage(this: &mut Self, #[comptime] config: G) -> Self::StageReader {
        ContinuousLoading::load_to_slice::<EG, ES, G>(
            &this.tensor_view,
            this.stage.as_slice_mut(),
            Ident::Lhs,
            config,
        );
        LhsReader::new(this.stage)
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, Ident::Lhs);
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, G: Config> Loader<EG, ES, G> for RhsTensorLoader<EG, ES, G> {
    type StageReader = RhsReader<ES, G::SmmConfig>;

    fn fill_stage(this: &mut Self, #[comptime] config: G) -> Self::StageReader {
        ContinuousLoading::load_to_slice::<EG, ES, G>(
            &this.tensor_view,
            this.stage.as_slice_mut(),
            Ident::Rhs,
            config,
        );
        RhsReader::new(this.stage)
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, Ident::Rhs);
    }
}
