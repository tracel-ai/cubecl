use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::block_info::{self, BlockInfo};
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
}

// #[cube]
// impl<E: Numeric> Loader<Line<E>> for LhsArrayLoader<E> {
//     type Gmem = Array<Line<E>>;
//     type BlockReader = LhsSmemTileReader<E, RowMajorTiling>;

//     fn new(gmem: Self::Gmem, layout: MatrixLayout, block_info: BlockInfo) -> Self {
//         LhsArrayLoader::<E> {
//             gmem,
//             smem: SharedMemory::<Line<E>>::new(comptime!(total_num_elements(block_info))),
//             gmem_layout: layout,
//             block_info,
//         }
//     }

//     fn fill_block(reader: &mut Self, _k_offset: u32) -> Self::BlockReader {
//         array_to_shared_memory(
//             &reader.gmem,
//             &mut reader.smem,
//             UNIT_POS_Y,
//             reader.block_info,
//         );

//         LhsSmemTileReader::<E, RowMajorTiling> {
//             smem: reader.smem,
//             block_info: reader.block_info,
//             _tiling_order: PhantomData::<RowMajorTiling>.runtime(),
//         }
//     }
// }

// #[cube]
// impl<E: Numeric> Loader<Line<E>> for RhsArrayLoader<E> {
//     type Gmem = Array<Line<E>>;
//     type BlockReader = RhsSmemTileReader<E, RowMajorTiling>;

//     fn new(gmem: Self::Gmem, layout: MatrixLayout, block_info: BlockInfo) -> Self {
//         RhsArrayLoader::<E> {
//             gmem,
//             smem: SharedMemory::<Line<E>>::new(comptime!(total_num_elements(block_info))),
//             gmem_layout: layout,
//             block_info,
//         }
//     }

//     // TODO bad api if k_offset not useful for all loaders
//     fn fill_block(reader: &mut Self, _k_offset: u32) -> Self::BlockReader {
//         array_to_shared_memory(
//             &reader.gmem,
//             &mut reader.smem,
//             UNIT_POS_Y,
//             reader.block_info,
//         );

//         RhsSmemTileReader::<E, RowMajorTiling> {
//             smem: reader.smem,
//             block_info: reader.block_info,
//             _tiling_order: PhantomData::<RowMajorTiling>.runtime(),
//         }
//     }
// }
