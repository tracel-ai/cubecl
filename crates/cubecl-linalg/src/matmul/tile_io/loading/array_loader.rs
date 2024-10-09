use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::block_info::{total_num_elements, BlockInfo};
use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::tile_io::loading::{array_to_shared_memory, RhsSmemTileReader};
use crate::matmul::tile_io::Loader;

use super::tiled_layout::RowMajorTiling;
use super::LhsSmemTileReader;

#[derive(CubeType)]
pub struct LhsArrayLoader<E: Numeric> {
    pub gmem: Array<Line<E>>,
    pub smem: SharedMemory<Line<E>>,
    pub gmem_layout: MatrixLayout,
    pub block_info: BlockInfo,
}

#[derive(CubeType)]
pub struct RhsArrayLoader<E: Numeric> {
    pub gmem: Array<Line<E>>,
    pub smem: SharedMemory<Line<E>>,
    pub gmem_layout: MatrixLayout,
    pub block_info: BlockInfo,
}

#[cube]
impl<E: Numeric> Loader<Line<E>> for LhsArrayLoader<E> {
    type Gmem = Array<Line<E>>;
    type BlockReader = LhsSmemTileReader<E, RowMajorTiling>;

    fn new(gmem: Self::Gmem, layout: MatrixLayout, block_info: BlockInfo) -> Self {
        LhsArrayLoader::<E> {
            gmem,
            smem: SharedMemory::<Line<E>>::new(comptime!(total_num_elements(block_info))),
            gmem_layout: layout,
            block_info,
        }
    }

    fn fill_block(reader: &mut Self, _k_offset: u32) -> Self::BlockReader {
        array_to_shared_memory(
            &reader.gmem,
            &mut reader.smem,
            UNIT_POS_Y,
            reader.block_info,
        );

        LhsSmemTileReader::<E, RowMajorTiling> {
            smem: reader.smem,
            block_info: reader.block_info,
            _tiling_order: PhantomData::<RowMajorTiling>.runtime(),
        }
    }
}

#[cube]
impl<E: Numeric> Loader<Line<E>> for RhsArrayLoader<E> {
    type Gmem = Array<Line<E>>;
    type BlockReader = RhsSmemTileReader<E, RowMajorTiling>;

    fn new(gmem: Self::Gmem, layout: MatrixLayout, block_info: BlockInfo) -> Self {
        RhsArrayLoader::<E> {
            gmem,
            smem: SharedMemory::<Line<E>>::new(comptime!(total_num_elements(block_info))),
            gmem_layout: layout,
            block_info,
        }
    }

    // TODO bad api if k_offset not useful for all loaders
    fn fill_block(reader: &mut Self, _k_offset: u32) -> Self::BlockReader {
        array_to_shared_memory(
            &reader.gmem,
            &mut reader.smem,
            UNIT_POS_Y,
            reader.block_info,
        );

        RhsSmemTileReader::<E, RowMajorTiling> {
            smem: reader.smem,
            block_info: reader.block_info,
            _tiling_order: PhantomData::<RowMajorTiling>.runtime(),
        }
    }
}
