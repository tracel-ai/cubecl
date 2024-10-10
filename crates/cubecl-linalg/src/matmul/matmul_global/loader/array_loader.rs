use super::Loader;
use crate::matmul::matmul_global::new_array_view;
use crate::matmul::matmul_global::ArrayView;
use crate::matmul::matmul_stage::Stage;
use crate::matmul::matmul_stage::{LhsStageReader, RhsStageReader};
use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::stage_info::StageInfo;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

#[derive(CubeType)]
pub struct LhsArrayLoader<E: Numeric, S: Stage<E>> {
    pub gmem_view: ArrayView<E>,
    pub stage: S,
}

#[derive(CubeType)]
pub struct RhsArrayLoader<E: Numeric, S: Stage<E>> {
    pub gmem_view: ArrayView<E>,
    pub stage: S,
}

#[cube]
impl<E: Numeric, S: Stage<E>> Loader<E> for LhsArrayLoader<E, S> {
    type GlobalView = ArrayView<E>;
    type StageReader = LhsStageReader<E, S>;

    fn new(
        array: Array<Line<E>>,
        #[comptime] layout: MatrixLayout,
        #[comptime] stage_info: StageInfo,
    ) -> Self {
        let line_size = array[0].size();
        let stage = S::new(layout, stage_info, line_size);
        let shape = (
            stage_info.num_tiles_x * stage_info.tile_size_x,
            stage_info.num_tiles_y * stage_info.tile_size_y,
        )
            .runtime();
        let gmem_view = new_array_view(array, layout, shape);

        LhsArrayLoader::<E, S> { gmem_view, stage }
    }

    fn fill_block(loader: &mut Self) -> Self::StageReader {
        S::fill::<E, Self::GlobalView>(&mut loader.stage, &loader.gmem_view);
        LhsStageReader::<E, S> {
            stage: loader.stage,
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
impl<E: Numeric, S: Stage<E>> Loader<E> for RhsArrayLoader<E, S> {
    type GlobalView = ArrayView<E>;
    type StageReader = RhsStageReader<E, S>;

    fn new(
        array: Array<Line<E>>,
        #[comptime] layout: MatrixLayout,
        #[comptime] stage_info: StageInfo,
    ) -> Self {
        let line_size = array[0].size();
        let stage = S::new(layout, stage_info, line_size);
        let shape = (
            stage_info.num_tiles_x * stage_info.tile_size_x,
            stage_info.num_tiles_y * stage_info.tile_size_y,
        )
            .runtime();
        let gmem_view = new_array_view(array, layout, shape);

        RhsArrayLoader::<E, S> { gmem_view, stage }
    }

    fn fill_block(loader: &mut Self) -> Self::StageReader {
        S::fill::<E, Self::GlobalView>(&mut loader.stage, &loader.gmem_view);
        RhsStageReader::<E, S> {
            stage: loader.stage,
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
