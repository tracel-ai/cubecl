use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::data::{Stage, Tile};
use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::tile_io::StageReader;

#[derive(CubeType)]
pub struct LhsBlockReader<E: Numeric, B: Stage<E>> {
    pub block: B,
    pub _e: PhantomData<E>,
}

#[derive(CubeType)]
pub struct RhsBlockReader<E: Numeric, B: Stage<E>> {
    pub block: B,
    pub _e: PhantomData<E>,
}

#[cube]
impl<E: Numeric, B: Stage<E>> StageReader<E> for LhsBlockReader<E, B> {
    fn read_tile(
        reader: &Self,
        compute_plane_offset: u32,
        buffer_offset: u32,
        _accumulator_offset: u32,
    ) -> Tile<'_, E> {
        B::get_tile(&reader.block, compute_plane_offset, buffer_offset)
    }

    fn slice_layout(reader: &Self) -> MatrixLayout {
        B::layout(&reader.block)
    }
}

#[cube]
impl<E: Numeric, B: Stage<E>> StageReader<E> for RhsBlockReader<E, B> {
    fn read_tile(
        reader: &Self,
        _compute_plane_offset: u32,
        buffer_offset: u32,
        accumulator_offset: u32,
    ) -> Tile<'_, E> {
        B::get_tile(&reader.block, buffer_offset, accumulator_offset)
    }

    fn slice_layout(reader: &Self) -> MatrixLayout {
        B::layout(&reader.block)
    }
}
