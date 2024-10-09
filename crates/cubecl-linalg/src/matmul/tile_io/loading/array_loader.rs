use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::block_info::BlockInfo;
use crate::matmul::data::{new_array_view, ArrayView, Block};
use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::tile_io::loading::RhsBlockReader;
use crate::matmul::tile_io::Loader;

use super::LhsBlockReader;

#[derive(CubeType)]
pub struct LhsArrayLoader<E: Numeric, B: Block<E>> {
    pub gmem_view: ArrayView<E>,
    pub block: B,
}

#[derive(CubeType)]
pub struct RhsArrayLoader<E: Numeric, B: Block<E>> {
    pub gmem_view: ArrayView<E>,
    pub block: B,
}

#[cube]
impl<E: Numeric, B: Block<E, GmemView = ArrayView<E>>> Loader<E> for LhsArrayLoader<E, B> {
    type GmemView = ArrayView<E>;
    type BlockReader = LhsBlockReader<E, B>;

    fn new(
        array: Array<Line<E>>,
        #[comptime] layout: MatrixLayout,
        #[comptime] block_info: BlockInfo,
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

    fn fill_block(loader: &mut Self) -> Self::BlockReader {
        B::fill(&mut loader.block, &loader.gmem_view);
        LhsBlockReader::<E, B> {
            block: loader.block,
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
impl<E: Numeric, B: Block<E, GmemView = ArrayView<E>>> Loader<E> for RhsArrayLoader<E, B> {
    type GmemView = ArrayView<E>;
    type BlockReader = RhsBlockReader<E, B>;

    fn new(
        array: Array<Line<E>>,
        #[comptime] layout: MatrixLayout,
        #[comptime] block_info: BlockInfo,
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

    fn fill_block(loader: &mut Self) -> Self::BlockReader {
        B::fill(&mut loader.block, &loader.gmem_view);
        RhsBlockReader::<E, B> {
            block: loader.block,
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
