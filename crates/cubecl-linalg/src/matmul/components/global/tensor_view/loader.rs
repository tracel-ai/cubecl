use crate::matmul::components::global::tensor_view::cyclic_loading::CyclicLoading;
use crate::matmul::components::global::{Config, Loader};
use crate::matmul::components::stage::{LhsReader, RhsReader, Stage};
use crate::matmul::components::Ident;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

use super::base::TensorView;

#[derive(CubeType)]
pub struct LhsLoader<EG: Numeric, ES: Numeric, G: Config> {
    pub tensor_view: TensorView<EG>,
    pub stage: Stage<ES>,
    pub _e: PhantomData<ES>,
    pub _config: PhantomData<G>,
}

#[derive(CubeType)]
pub struct RhsLoader<EG: Numeric, ES: Numeric, G: Config> {
    pub tensor_view: TensorView<EG>,
    pub stage: Stage<ES>,
    pub _e: PhantomData<ES>,
    pub _config: PhantomData<G>,
}

#[cube]
impl<EG: Numeric, ES: Numeric, G: Config> LhsLoader<EG, ES, G> {
    pub fn new(
        tensor: Tensor<Line<EG>>,
        x_offset: u32,
        y_offset: u32,
        nth_batch: u32,
        #[comptime] config: G,
    ) -> Self {
        let stage = Stage::new::<G::SmmConfig>(Ident::Lhs, config.to_smm_config());
        let tensor_view = TensorView::new(tensor, x_offset, y_offset, nth_batch);

        LhsLoader::<EG, ES, G> {
            tensor_view,
            stage,
            _e: PhantomData::<ES>.runtime(),
            _config: PhantomData::<G>.runtime(),
        }
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, G: Config> RhsLoader<EG, ES, G> {
    pub fn new(
        tensor: Tensor<Line<EG>>,
        x_offset: u32,
        y_offset: u32,
        nth_batch: u32,
        #[comptime] config: G,
    ) -> Self {
        let stage = Stage::new::<G::SmmConfig>(Ident::Rhs, config.to_smm_config());
        let tensor_view = TensorView::new(tensor, x_offset, y_offset, nth_batch);

        RhsLoader::<EG, ES, G> {
            tensor_view,
            stage,
            _e: PhantomData::<ES>.runtime(),
            _config: PhantomData::<G>.runtime(),
        }
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, G: Config> Loader<EG, ES, G> for LhsLoader<EG, ES, G> {
    type StageReader = LhsReader<ES, G::SmmConfig>;

    fn fill_stage(this: &mut Self, #[comptime] config: G) -> Self::StageReader {
        CyclicLoading::load_to_slice::<EG, ES, G>(
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
impl<EG: Numeric, ES: Numeric, G: Config> Loader<EG, ES, G> for RhsLoader<EG, ES, G> {
    type StageReader = RhsReader<ES, G::SmmConfig>;

    fn fill_stage(this: &mut Self, #[comptime] config: G) -> Self::StageReader {
        CyclicLoading::load_to_slice::<EG, ES, G>(
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
