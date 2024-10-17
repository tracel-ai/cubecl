use crate::matmul::matmul_stage::Stage;
use crate::matmul::matrix_layout::MatrixLayout;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

#[cube]
pub trait StageReader<ES: Numeric>: CubeType {
    fn read_tile(
        stage_reader: &Self,
        compute_plane_offset: u32,
        buffer_offset: u32,
        accumulator_offset: u32,
    ) -> (&Slice<'_, Line<ES>>, MatrixLayout);

    // Maybe delete if we don't need layout prior to slice, or if available in config
    fn slice_layout(stage_reader: &Self) -> MatrixLayout;
}

#[derive(CubeType)]
pub struct LhsStageReader<ES: Numeric, S: Stage<ES>> {
    pub stage: S,
    pub _e: PhantomData<ES>,
}

#[derive(CubeType)]
pub struct RhsStageReader<ES: Numeric, S: Stage<ES>> {
    pub stage: S,
    pub _e: PhantomData<ES>,
}

#[cube]
impl<ES: Numeric, S: Stage<ES>> StageReader<ES> for LhsStageReader<ES, S> {
    fn read_tile(
        reader: &Self,
        compute_plane_offset: u32,
        buffer_offset: u32,
        _accumulator_offset: u32,
    ) -> (&Slice<'_, Line<ES>>, MatrixLayout) {
        S::get_tile(&reader.stage, compute_plane_offset, buffer_offset)
    }

    fn slice_layout(reader: &Self) -> MatrixLayout {
        S::layout(&reader.stage)
    }
}

#[cube]
impl<ES: Numeric, S: Stage<ES>> StageReader<ES> for RhsStageReader<ES, S> {
    fn read_tile(
        reader: &Self,
        _compute_plane_offset: u32,
        buffer_offset: u32,
        accumulator_offset: u32,
    ) -> (&Slice<'_, Line<ES>>, MatrixLayout) {
        S::get_tile(&reader.stage, buffer_offset, accumulator_offset)
    }

    fn slice_layout(reader: &Self) -> MatrixLayout {
        S::layout(&reader.stage)
    }
}
