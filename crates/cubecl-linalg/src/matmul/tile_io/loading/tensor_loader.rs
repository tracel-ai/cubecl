use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::block_info::BlockInfo;
use crate::matmul::data::{new_tensor_view, Block, GmemView, TensorView};
use crate::matmul::matrix_layout::MatrixLayout;
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

    fn new(
        tensor: Tensor<Line<E>>,
        #[comptime] layout: MatrixLayout,
        #[comptime] block_info: BlockInfo,
    ) -> Self {
        let line_size = comptime!(tensor.line_size());
        let block = B::new(layout, block_info, line_size);
        let gmem_view = new_tensor_view(tensor, layout);

        LhsTensorLoader::<E, B> { gmem_view, block }
    }

    fn fill_block(loader: &mut Self) -> Self::BlockReader {
        B::fill(&mut loader.block, &loader.gmem_view);
        LhsBlockReader::<E, B> {
            block: loader.block,
            _e: PhantomData::<E>.runtime(),
        }
    }

    fn advance(loader: &mut Self, k_offset: u32) {
        TensorView::update_view(&mut loader.gmem_view, 0, k_offset);
    }
}

#[cube]
impl<E: Numeric, B: Block<E, GmemView = TensorView<E>>> Loader<E> for RhsTensorLoader<E, B> {
    type GmemView = TensorView<E>;
    type BlockReader = RhsBlockReader<E, B>;

    fn new(
        tensor: Tensor<Line<E>>,
        #[comptime] layout: MatrixLayout,
        #[comptime] block_info: BlockInfo,
    ) -> Self {
        let line_size = comptime!(tensor.line_size());
        let block = B::new(layout, block_info, line_size);
        let gmem_view = new_tensor_view(tensor, layout);

        RhsTensorLoader::<E, B> { gmem_view, block }
    }

    fn fill_block(loader: &mut Self) -> Self::BlockReader {
        B::fill(&mut loader.block, &loader.gmem_view);
        RhsBlockReader::<E, B> {
            block: loader.block,
            _e: PhantomData::<E>.runtime(),
        }
    }

    fn advance(loader: &mut Self, k_offset: u32) {
        TensorView::update_view(&mut loader.gmem_view, k_offset, 0);
    }
}
