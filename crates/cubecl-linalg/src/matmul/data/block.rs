use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::block_info::{tile_num_elements, total_num_elements};
use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::tile_io::loading::tiled_layout::TilingOrder;
use crate::matmul::tile_io::loading::Tensor2Smem;
use crate::matmul::{block_info::BlockInfo, data::new_tile};

use super::{GmemView, TensorView, Tile};

#[cube]
/// Blocks are created with a filled shared memory,
/// then can give out tiles
/// When shared memory changes, a new block should be made
///
/// Double buffering: we could have two blocks, sharing a SharedMemory
pub trait Block<E: Numeric>: CubeType + Clone + Copy + IntoRuntime + Send + Sync + 'static {
    type GmemView: GmemView<E>;
    type Smem: CubeType;

    fn new(
        layout: MatrixLayout,
        #[comptime] block_info: BlockInfo,
        #[comptime] line_size: u32,
    ) -> Self;
    fn fill(block: &mut Self, gmem: &Self::GmemView);
    fn get_tile(block: &Self, x: u32, y: u32) -> Tile<'_, E>;
    /// Hopefully delete
    fn layout(block: &Self) -> MatrixLayout;
}

#[derive(CubeType, Clone, Copy)]
pub struct SharedMemoryBlock<E: Numeric, O: TilingOrder, TS: Tensor2Smem> {
    smem: SharedMemory<Line<E>>,
    layout: MatrixLayout,
    block_info: BlockInfo,
    _tiling_order: PhantomData<O>,
    _tensor_2_smem: PhantomData<TS>,
}

#[cube]
impl<E: Numeric, O: TilingOrder, TS: Tensor2Smem> Block<E> for SharedMemoryBlock<E, O, TS> {
    type GmemView = TensorView<E>;
    type Smem = SharedMemory<Line<E>>;

    fn new(
        layout: MatrixLayout,
        #[comptime] block_info: BlockInfo,
        #[comptime] line_size: u32,
    ) -> Self {
        let smem = SharedMemory::new_lined(
            comptime!(total_num_elements(block_info) / line_size),
            line_size,
        );

        SharedMemoryBlock::<E, O, TS> {
            smem,
            layout,
            block_info: block_info.runtime(),
            _tiling_order: PhantomData::<O>.runtime(),
            _tensor_2_smem: PhantomData::<TS>.runtime(),
        }
    }

    fn fill(block: &mut Self, gmem: &Self::GmemView) {
        TS::tensor_to_shared_memory::<E, E, O>(gmem, &mut block.smem, block.block_info)
    }

    fn get_tile(block: &Self, x: u32, y: u32) -> Tile<'_, E> {
        let tile_stride = tile_num_elements(block.block_info);

        let nth_tile = O::to_nth_tile(
            x,
            y,
            block.block_info.num_tiles_x,
            block.block_info.num_tiles_y,
        );

        let start = nth_tile * tile_stride;
        new_tile(
            x,
            y,
            block.smem.slice(start, start + tile_stride),
            block.layout,
        )
    }

    fn layout(block: &Self) -> MatrixLayout {
        block.layout
    }
}
