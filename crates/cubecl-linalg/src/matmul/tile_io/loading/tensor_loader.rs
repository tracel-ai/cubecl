use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::block_info::BlockInfo;
use crate::matmul::data::{Block, TensorView};
use crate::matmul::tile_io::Loader;

use super::{LhsBlockReader, RhsBlockReader};

#[derive(CubeType)]
pub struct LhsTensorLoader<E: Numeric, B: Block<E>> {
    pub gmem_view: TensorView<E>,
    pub block: B,
}

#[derive(CubeType)]
pub struct RhsTensorLoader<E: Numeric, B: Block<E>> {
    pub gmem_view: TensorView<E>,
    pub block: B,
}

#[cube]
impl<E: Numeric, B: Block<E, GmemView = TensorView<E>>> Loader<E> for LhsTensorLoader<E, B> {
    type GmemView = TensorView<E>;
    type BlockReader = LhsBlockReader<E, B>;

    fn new(gmem_view: Self::GmemView, #[comptime] block_info: BlockInfo) -> Self {
        let line_size = comptime!(gmem_view.tensor.line_size());
        let block = B::new(gmem_view.layout, block_info, line_size);

        LhsTensorLoader::<E, B> { gmem_view, block }
    }

    fn fill_block(loader: &mut Self) -> Self::BlockReader {
        B::fill(&mut loader.block, &loader.gmem_view);
        LhsBlockReader::<E, B> {
            block: loader.block,
            _e: PhantomData::<E>.runtime(),
        }
    }
}

#[cube]
impl<E: Numeric, B: Block<E, GmemView = TensorView<E>>> Loader<E> for RhsTensorLoader<E, B> {
    type GmemView = TensorView<E>;
    type BlockReader = RhsBlockReader<E, B>;

    fn new(gmem_view: Self::GmemView, #[comptime] block_info: BlockInfo) -> Self {
        let line_size = comptime!(gmem_view.tensor.line_size());
        let block = B::new(gmem_view.layout, block_info, line_size);

        RhsTensorLoader::<E, B> { gmem_view, block }
    }

    fn fill_block(loader: &mut Self) -> Self::BlockReader {
        B::fill(&mut loader.block, &loader.gmem_view);
        RhsBlockReader::<E, B> {
            block: loader.block,
            _e: PhantomData::<E>.runtime(),
        }
    }
}
