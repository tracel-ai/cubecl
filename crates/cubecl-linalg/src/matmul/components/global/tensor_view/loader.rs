use crate::matmul::components::global::tensor_view::cyclic_loading::CyclicLoading;
use crate::matmul::components::global::Loader;
use crate::matmul::components::stage::{LhsReader, RhsReader, Stage};
use crate::matmul::components::{global, Ident};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::base::TensorView;

#[derive(CubeType)]
pub struct LhsLoader<EG: Numeric, ES: Numeric> {
    pub tensor_view: TensorView<EG>,
    pub stage: Stage<ES>,
}

#[derive(CubeType)]
pub struct RhsLoader<EG: Numeric, ES: Numeric> {
    pub tensor_view: TensorView<EG>,
    pub stage: Stage<ES>,
}

#[cube]
impl<EG: Numeric, ES: Numeric> LhsLoader<EG, ES> {
    pub fn new<G: global::Config>(
        tensor: &Tensor<Line<EG>>,
        x_offset: u32,
        y_offset: u32,
        nth_batch: u32,
        #[comptime] config: G,
    ) -> Self {
        let stage = Stage::new::<G::SmmConfig>(Ident::Lhs, config.to_smm_config());
        let tensor_view = TensorView::new(tensor, x_offset, y_offset, nth_batch);

        LhsLoader::<EG, ES> { tensor_view, stage }
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric> RhsLoader<EG, ES> {
    pub fn new<G: global::Config>(
        tensor: &Tensor<Line<EG>>,
        x_offset: u32,
        y_offset: u32,
        nth_batch: u32,
        #[comptime] config: G,
    ) -> Self {
        let stage = Stage::new::<G::SmmConfig>(Ident::Rhs, config.to_smm_config());
        let tensor_view = TensorView::new(tensor, x_offset, y_offset, nth_batch);

        RhsLoader::<EG, ES> { tensor_view, stage }
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric> Loader<EG, ES> for LhsLoader<EG, ES> {
    type StageReader = LhsReader<ES>;

    fn fill_stage<G: global::Config>(this: &mut Self, #[comptime] config: G) -> Self::StageReader {
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
impl<EG: Numeric, ES: Numeric> Loader<EG, ES> for RhsLoader<EG, ES> {
    type StageReader = RhsReader<ES>;

    fn fill_stage<G: global::Config>(this: &mut Self, #[comptime] config: G) -> Self::StageReader {
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
