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
pub struct LhsArrayLoader<EG: Numeric, ES: Numeric, S: Stage<ES>> {
    pub gmem_view: ArrayView<EG>,
    pub stage: S,
    pub _e: PhantomData<ES>,
}

#[derive(CubeType)]
pub struct RhsArrayLoader<EG: Numeric, ES: Numeric, S: Stage<ES>> {
    pub gmem_view: ArrayView<EG>,
    pub stage: S,
    pub _e: PhantomData<ES>,
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: Stage<ES>> Loader<EG, ES> for LhsArrayLoader<EG, ES, S> {
    type GlobalView = ArrayView<EG>;
    type StageReader = LhsStageReader<ES, S>;

    fn new(
        array: Array<Line<EG>>,
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

        LhsArrayLoader::<EG, ES, S> {
            gmem_view,
            stage,
            _e: PhantomData::<ES>.runtime(),
        }
    }

    fn fill_block(loader: &mut Self) -> Self::StageReader {
        S::fill::<EG, Self::GlobalView>(&mut loader.stage, &loader.gmem_view);
        LhsStageReader::<ES, S> {
            stage: loader.stage,
            _e: PhantomData::<ES>.runtime(),
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
impl<EG: Numeric, ES: Numeric, S: Stage<ES>> Loader<EG, ES> for RhsArrayLoader<EG, ES, S> {
    type GlobalView = ArrayView<EG>;
    type StageReader = RhsStageReader<ES, S>;

    fn new(
        array: Array<Line<EG>>,
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

        RhsArrayLoader::<EG, ES, S> {
            gmem_view,
            stage,
            _e: PhantomData::<ES>.runtime(),
        }
    }

    fn fill_block(loader: &mut Self) -> Self::StageReader {
        S::fill::<EG, Self::GlobalView>(&mut loader.stage, &loader.gmem_view);
        RhsStageReader::<ES, S> {
            stage: loader.stage,
            _e: PhantomData::<ES>.runtime(),
        }
    }

    fn init_view(_loader: &mut Self, _cube_offset: u32, _k_start: u32) {
        // Array loader does not support offsets
    }

    fn advance_view(_loader: &mut Self, _k_offset: u32) {
        // Array loader does not support offsets
    }
}
