use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::data::{new_array_view, ArrayView, Stage};
use crate::matmul::matmul_stage::{LhsStageReader, RhsStageReader};
use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::stage_info::StageInfo;

use super::Loader;

#[derive(CubeType)]
pub struct LhsArrayLoader<E: Numeric, B: Stage<E>> {
    pub gmem_view: ArrayView<E>,
    pub block: B,
}

#[derive(CubeType)]
pub struct RhsArrayLoader<E: Numeric, B: Stage<E>> {
    pub gmem_view: ArrayView<E>,
    pub block: B,
}

#[cube]
impl<E: Numeric, B: Stage<E>> Loader<E> for LhsArrayLoader<E, B> {
    type GlobalView = ArrayView<E>;
    type StageReader = LhsStageReader<E, B>;

    fn new(
        array: Array<Line<E>>,
        #[comptime] layout: MatrixLayout,
        #[comptime] block_info: StageInfo,
    ) -> Self {
        let line_size = array[0].size();
        let block = B::new(layout, block_info, line_size);
        let shape = (
            block_info.num_tiles_x * block_info.tile_size_x,
            block_info.num_tiles_y * block_info.tile_size_y,
        )
            .runtime();
        let gmem_view = new_array_view(array, layout, shape);

        LhsArrayLoader::<E, B> { gmem_view, block }
    }

    fn fill_block(loader: &mut Self) -> Self::StageReader {
        B::fill::<E, Self::GlobalView>(&mut loader.block, &loader.gmem_view);
        LhsStageReader::<E, B> {
            stage: loader.block,
            _e: PhantomData::<E>.runtime(),
        }
    }

    fn init_view(_loader: &mut Self, _cube_offset: u32, _k_start: u32) {
        // Array loader does not support offsets
    }

    fn advance_view(_loader: &mut Self, _k_offset: u32) {
        // Array loader does not support offsets
    }
}

#[cube]
impl<E: Numeric, B: Stage<E>> Loader<E> for RhsArrayLoader<E, B> {
    type GlobalView = ArrayView<E>;
    type StageReader = RhsStageReader<E, B>;

    fn new(
        array: Array<Line<E>>,
        #[comptime] layout: MatrixLayout,
        #[comptime] block_info: StageInfo,
    ) -> Self {
        let line_size = array[0].size();
        let block = B::new(layout, block_info, line_size);
        let shape = (
            block_info.num_tiles_x * block_info.tile_size_x,
            block_info.num_tiles_y * block_info.tile_size_y,
        )
            .runtime();
        let gmem_view = new_array_view(array, layout, shape);

        RhsArrayLoader::<E, B> { gmem_view, block }
    }

    fn fill_block(loader: &mut Self) -> Self::StageReader {
        B::fill::<E, Self::GlobalView>(&mut loader.block, &loader.gmem_view);
        RhsStageReader::<E, B> {
            stage: loader.block,
            _e: PhantomData::<E>.runtime(),
        }
    }

    fn init_view(_loader: &mut Self, _cube_offset: u32, _k_start: u32) {
        // Array loader does not support offsets
    }

    fn advance_view(_loader: &mut Self, _k_offset: u32) {
        // Array loader does not support offsets
    }
}
