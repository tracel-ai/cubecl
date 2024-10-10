use crate::matmul::matmul_instruction::Tile;
use crate::matmul::matmul_stage::Stage;
use crate::matmul::matrix_layout::MatrixLayout;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

#[cube]
pub trait StageReader<E: Numeric>: CubeType {
    fn read_tile(
        tile_reader: &Self,
        compute_plane_offset: u32,
        buffer_offset: u32,
        accumulator_offset: u32,
    ) -> Tile<'_, E>;

    // Maybe delete if we don't need layout prior to slice
    fn slice_layout(tile_reader: &Self) -> MatrixLayout;
}

#[derive(CubeType)]
pub struct LhsStageReader<E: Numeric, B: Stage<E>> {
    pub stage: B,
    pub _e: PhantomData<E>,
}

#[derive(CubeType)]
pub struct RhsStageReader<E: Numeric, B: Stage<E>> {
    pub stage: B,
    pub _e: PhantomData<E>,
}

#[cube]
impl<E: Numeric, B: Stage<E>> StageReader<E> for LhsStageReader<E, B> {
    fn read_tile(
        reader: &Self,
        compute_plane_offset: u32,
        buffer_offset: u32,
        _accumulator_offset: u32,
    ) -> Tile<'_, E> {
        B::get_tile(&reader.stage, compute_plane_offset, buffer_offset)
    }

    fn slice_layout(reader: &Self) -> MatrixLayout {
        B::layout(&reader.stage)
    }
}

#[cube]
impl<E: Numeric, B: Stage<E>> StageReader<E> for RhsStageReader<E, B> {
    fn read_tile(
        reader: &Self,
        _compute_plane_offset: u32,
        buffer_offset: u32,
        accumulator_offset: u32,
    ) -> Tile<'_, E> {
        B::get_tile(&reader.stage, buffer_offset, accumulator_offset)
    }

    fn slice_layout(reader: &Self) -> MatrixLayout {
        B::layout(&reader.stage)
    }
}
