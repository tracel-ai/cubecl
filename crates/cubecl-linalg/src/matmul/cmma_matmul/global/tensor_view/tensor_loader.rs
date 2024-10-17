use crate::matmul::cmma_matmul::global::new_tensor_view;
use crate::matmul::cmma_matmul::stage::{LhsStageReader, RhsStageReader, Stage};
use crate::matmul::matmul_global::{Loader, ReadView};
use crate::matmul::matrix::Ident;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

use super::TensorView;

#[derive(CubeType)]
pub struct LhsTensorLoader<EG: Numeric, ES: Numeric> {
    pub gmem_view: TensorView<EG>,
    pub stage: Stage,
    pub _e: PhantomData<ES>,
}

#[derive(CubeType)]
pub struct RhsTensorLoader<EG: Numeric, ES: Numeric> {
    pub gmem_view: TensorView<EG>,
    pub stage: S,
    pub _e: PhantomData<ES>,
}

#[cube]
impl<EG: Numeric, ES: Numeric> Loader<EG, ES> for LhsTensorLoader<EG, ES> {
    type ReadView = TensorView<EG>;
    type StageReader = LhsStageReader<ES>;

    fn new(tensor: Tensor<Line<EG>>, #[comptime] config: Self::Config) -> Self {
        let stage = S::new(Ident::Lhs, config);
        let gmem_view = new_tensor_view(tensor);

        LhsTensorLoader::<EG, ES, S> {
            gmem_view,
            stage,
            _e: PhantomData::<ES>.runtime(),
        }
    }

    fn fill_stage(self_: &mut Self, config: StageCmmaMatmulConfig) -> Self::StageReader {
        S::fill::<EG, Self::ReadView>(&mut self_.stage, &self_.gmem_view, Ident::Lhs, config);
        LhsStageReader::<ES, S> {
            stage: self_.stage,
            _e: PhantomData::<ES>.runtime(),
        }
    }

    fn init_view(self_: &mut Self, cube_offset: u32, k_start: u32) {
        TensorView::init_view(&mut self_.gmem_view, cube_offset, k_start);
    }

    fn advance_view(self_: &mut Self, k_offset: u32) {
        TensorView::update_view(&mut self_.gmem_view, 0, k_offset);
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: Stage<ES, Config = CmmaConfig>> Loader<EG, ES>
    for RhsTensorLoader<EG, ES, S>
{
    type ReadView = TensorView<EG>;
    type StageReader = RhsStageReader<ES, S>;
    type Config = CmmaConfig;

    fn new(tensor: Tensor<Line<EG>>, #[comptime] config: Self::Config) -> Self {
        let stage = S::new(Ident::Rhs, config);
        let gmem_view = new_tensor_view(tensor);

        RhsTensorLoader::<EG, ES, S> {
            gmem_view,
            stage,
            _e: PhantomData::<ES>.runtime(),
        }
    }

    fn fill_stage(self_: &mut Self, config: Self::Config) -> Self::StageReader {
        S::fill::<EG, Self::ReadView>(&mut self_.stage, &self_.gmem_view, Ident::Rhs, config);
        RhsStageReader::<ES, S> {
            stage: self_.stage,
            _e: PhantomData::<ES>.runtime(),
        }
    }

    fn init_view(self_: &mut Self, cube_offset: u32, k_start: u32) {
        TensorView::init_view(&mut self_.gmem_view, k_start, cube_offset);
    }

    fn advance_view(self_: &mut Self, k_offset: u32) {
        TensorView::update_view(&mut self_.gmem_view, k_offset, 0);
    }
}
