use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::data::{new_tensor_view, GlobalView, Stage, TensorView};
use crate::matmul::matmul_stage::{LhsStageReader, RhsStageReader};
use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::stage_info::StageInfo;

use super::Loader;

#[derive(CubeType)]
pub struct LhsTensorLoader<E: Numeric, S: Stage<E>> {
    pub gmem_view: TensorView<E>,
    pub stage: S,
}

#[derive(CubeType)]
pub struct RhsTensorLoader<E: Numeric, S: Stage<E>> {
    pub gmem_view: TensorView<E>,
    pub stage: S,
}

#[cube]
impl<E: Numeric, S: Stage<E>> Loader<E> for LhsTensorLoader<E, S> {
    type GlobalView = TensorView<E>;
    type StageReader = LhsStageReader<E, S>;

    fn new(
        tensor: Tensor<Line<E>>,
        #[comptime] layout: MatrixLayout,
        #[comptime] stage: StageInfo,
    ) -> Self {
        let line_size = comptime!(tensor.line_size());
        let stage = S::new(layout, stage, line_size);
        let gmem_view = new_tensor_view(tensor, layout);

        LhsTensorLoader::<E, S> { gmem_view, stage }
    }

    fn fill_block(loader: &mut Self) -> Self::StageReader {
        S::fill::<E, Self::GlobalView>(&mut loader.stage, &loader.gmem_view);
        LhsStageReader::<E, S> {
            stage: loader.stage,
            _e: PhantomData::<E>.runtime(),
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
impl<E: Numeric, S: Stage<E>> Loader<E> for RhsTensorLoader<E, S> {
    type GlobalView = TensorView<E>;
    type StageReader = RhsStageReader<E, S>;

    fn new(
        tensor: Tensor<Line<E>>,
        #[comptime] layout: MatrixLayout,
        #[comptime] stage_info: StageInfo,
    ) -> Self {
        let line_size = comptime!(tensor.line_size());
        let stage = S::new(layout, stage_info, line_size);
        let gmem_view = new_tensor_view(tensor, layout);

        RhsTensorLoader::<E, S> { gmem_view, stage }
    }

    fn fill_block(loader: &mut Self) -> Self::StageReader {
        S::fill::<E, Self::GlobalView>(&mut loader.stage, &loader.gmem_view);
        RhsStageReader::<E, S> {
            stage: loader.stage,
            _e: PhantomData::<E>.runtime(),
        }
    }

    fn init_view(loader: &mut Self, cube_offset: u32, k_start: u32) {
        TensorView::init_view(&mut loader.gmem_view, k_start, cube_offset);
    }

    fn advance_view(loader: &mut Self, k_offset: u32) {
        TensorView::update_view(&mut loader.gmem_view, k_offset, 0);
    }
}
