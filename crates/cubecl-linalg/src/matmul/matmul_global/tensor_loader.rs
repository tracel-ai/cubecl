use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::data::{new_tensor_view, GlobalView, Stage, TensorView};
use crate::matmul::matmul_stage::{LhsStageReader, RhsStageReader, StageReader};
use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::stage_info::StageInfo;

#[cube]
pub trait Loader<E: Numeric>: CubeType + 'static + Send + Sync {
    type GlobalView: GlobalView<E>;
    type StageReader: StageReader<E>;

    fn new(
        gmem: <Self::GlobalView as GlobalView<E>>::Global,
        #[comptime] layout: MatrixLayout,
        #[comptime] block_info: StageInfo,
    ) -> Self;

    fn fill_block(loader: &mut Self) -> Self::StageReader;

    fn init_view(loader: &mut Self, cube_offset: u32, k_start: u32);

    fn advance_view(loader: &mut Self, k_offset: u32);
}

#[derive(CubeType)]
pub struct LhsTensorLoader<E: Numeric, B: Stage<E>> {
    pub gmem_view: TensorView<E>,
    pub block: B,
}

#[derive(CubeType)]
pub struct RhsTensorLoader<E: Numeric, B: Stage<E>> {
    pub gmem_view: TensorView<E>,
    pub block: B,
}

#[cube]
impl<E: Numeric, B: Stage<E>> Loader<E> for LhsTensorLoader<E, B> {
    type GlobalView = TensorView<E>;
    type StageReader = LhsStageReader<E, B>;

    fn new(
        tensor: Tensor<Line<E>>,
        #[comptime] layout: MatrixLayout,
        #[comptime] block_info: StageInfo,
    ) -> Self {
        let line_size = comptime!(tensor.line_size());
        let block = B::new(layout, block_info, line_size);
        let gmem_view = new_tensor_view(tensor, layout);

        LhsTensorLoader::<E, B> { gmem_view, block }
    }

    fn fill_block(loader: &mut Self) -> Self::StageReader {
        B::fill::<E, Self::GlobalView>(&mut loader.block, &loader.gmem_view);
        LhsStageReader::<E, B> {
            stage: loader.block,
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
impl<E: Numeric, B: Stage<E>> Loader<E> for RhsTensorLoader<E, B> {
    type GlobalView = TensorView<E>;
    type StageReader = RhsStageReader<E, B>;

    fn new(
        tensor: Tensor<Line<E>>,
        #[comptime] layout: MatrixLayout,
        #[comptime] block_info: StageInfo,
    ) -> Self {
        let line_size = comptime!(tensor.line_size());
        let block = B::new(layout, block_info, line_size);
        let gmem_view = new_tensor_view(tensor, layout);

        RhsTensorLoader::<E, B> { gmem_view, block }
    }

    fn fill_block(loader: &mut Self) -> Self::StageReader {
        B::fill::<E, Self::GlobalView>(&mut loader.block, &loader.gmem_view);
        RhsStageReader::<E, B> {
            stage: loader.block,
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
