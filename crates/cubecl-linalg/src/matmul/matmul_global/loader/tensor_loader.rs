use crate::matmul::cmma_matmul::config::CmmaConfig;
use crate::matmul::matmul_global::new_tensor_view;
use crate::matmul::matmul_global::GlobalView;
use crate::matmul::matmul_global::TensorView;
use crate::matmul::matmul_stage::Stage;
use crate::matmul::matmul_stage::{LhsStageReader, RhsStageReader};
use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::stage_info::StageInfo;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

use super::Loader;

#[derive(CubeType)]
pub struct LhsTensorLoader<EG: Numeric, ES: Numeric, S: Stage<ES>> {
    pub gmem_view: TensorView<EG>,
    pub stage: S,
    pub _e: PhantomData<ES>,
}

#[derive(CubeType)]
pub struct RhsTensorLoader<EG: Numeric, ES: Numeric, S: Stage<ES>> {
    pub gmem_view: TensorView<EG>,
    pub stage: S,
    pub _e: PhantomData<ES>,
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: Stage<ES>> Loader<EG, ES> for LhsTensorLoader<EG, ES, S> {
    type GlobalView = TensorView<EG>;
    type StageReader = LhsStageReader<ES, S>;
    type Config = CmmaConfig;

    fn new(
        tensor: Tensor<Line<EG>>,
        #[comptime] layout: MatrixLayout,
        #[comptime] stage: StageInfo,
    ) -> Self {
        let line_size = comptime!(tensor.line_size());
        let stage = S::new(layout, stage, line_size);
        let gmem_view = new_tensor_view(tensor, layout);

        LhsTensorLoader::<EG, ES, S> {
            gmem_view,
            stage,
            _e: PhantomData::<ES>.runtime(),
        }
    }

    fn fill_stage(loader: &mut Self, config: Self::Config) -> Self::StageReader {
        S::fill::<EG, Self::GlobalView>(&mut loader.stage, &loader.gmem_view, config);
        LhsStageReader::<ES, S> {
            stage: loader.stage,
            _e: PhantomData::<ES>.runtime(),
        }
    }

    fn init_view(loader: &mut Self, cube_offset: u32, k_start: u32) {
        TensorView::init_view(&mut loader.gmem_view, cube_offset, k_start);
    }

    fn advance_view(loader: &mut Self, k_offset: u32) {
        TensorView::update_view(&mut loader.gmem_view, 0, k_offset);
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: Stage<ES>> Loader<EG, ES> for RhsTensorLoader<EG, ES, S> {
    type GlobalView = TensorView<EG>;
    type StageReader = RhsStageReader<ES, S>;
    type Config = CmmaConfig;

    fn new(
        tensor: Tensor<Line<EG>>,
        #[comptime] layout: MatrixLayout,
        #[comptime] stage_info: StageInfo,
    ) -> Self {
        let line_size = comptime!(tensor.line_size());
        let stage = S::new(layout, stage_info, line_size);
        let gmem_view = new_tensor_view(tensor, layout);

        RhsTensorLoader::<EG, ES, S> {
            gmem_view,
            stage,
            _e: PhantomData::<ES>.runtime(),
        }
    }

    fn fill_stage(loader: &mut Self, config: Self::Config) -> Self::StageReader {
        S::fill::<EG, Self::GlobalView>(&mut loader.stage, &loader.gmem_view, config);
        RhsStageReader::<ES, S> {
            stage: loader.stage,
            _e: PhantomData::<ES>.runtime(),
        }
    }

    fn init_view(loader: &mut Self, cube_offset: u32, k_start: u32) {
        TensorView::init_view(&mut loader.gmem_view, k_start, cube_offset);
    }

    fn advance_view(loader: &mut Self, k_offset: u32) {
        TensorView::update_view(&mut loader.gmem_view, k_offset, 0);
    }
}
