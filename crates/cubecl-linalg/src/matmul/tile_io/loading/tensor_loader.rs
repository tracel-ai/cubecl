use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::block_info::{total_num_elements, BlockInfo};
use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::tile_io::loading::Tensor2Smem;
use crate::matmul::tile_io::loading::Tensor2SmemContinuous;
use crate::matmul::tile_io::loading::{LhsSmemTileReader, RhsSmemTileReader};
use crate::matmul::tile_io::Loader;

use super::tiled_layout::TilingOrder;

#[derive(CubeType)]
pub struct LhsTensorLoader<E: Numeric, T: TilingOrder> {
    pub gmem: Tensor<Line<E>>,
    pub smem: SharedMemory<Line<E>>,
    pub gmem_layout: MatrixLayout,
    pub cube_offset: u32,
    pub block_info: BlockInfo,
    pub _tiling_order: PhantomData<T>,
}

#[derive(CubeType)]
pub struct RhsTensorLoader<E: Numeric, T: TilingOrder> {
    pub gmem: Tensor<Line<E>>,
    pub smem: SharedMemory<Line<E>>,
    pub gmem_layout: MatrixLayout,
    pub cube_offset: u32,
    pub block_info: BlockInfo,
    pub _tiling_order: PhantomData<T>,
}

#[cube]
impl<E: Numeric, T: TilingOrder> Loader<Line<E>> for LhsTensorLoader<E, T> {
    type Gmem = Tensor<Line<E>>;
    type TileReader = LhsSmemTileReader<E, T>;

    fn new(
        gmem: Self::Gmem,
        gmem_layout: MatrixLayout,
        #[comptime] block_info: BlockInfo,
    ) -> LhsTensorLoader<E, T> {
        let line_size = gmem.line_size();
        let smem = SharedMemory::new_lined(
            comptime!(total_num_elements(block_info) / line_size),
            line_size,
        );

        LhsTensorLoader::<E, T> {
            gmem,
            smem,
            gmem_layout,
            cube_offset: CUBE_POS_X,
            block_info: block_info.runtime(),
            _tiling_order: PhantomData::<T>.runtime(),
        }
    }

    fn load_block(reader: &mut Self, k_offset: u32) -> Self::TileReader {
        // Assuming RowMajor layout
        // TODO refactor
        let is_row_major = true;
        let (gmem_row_offset, gmem_col_offset) = match is_row_major {
            true => (reader.cube_offset, k_offset),
            false => (k_offset, reader.cube_offset),
        };

        Tensor2SmemContinuous::tensor_to_shared_memory(
            &reader.gmem,
            &mut reader.smem,
            gmem_row_offset,
            gmem_col_offset,
            reader.block_info,
        );

        LhsSmemTileReader::<E, T> {
            smem: reader.smem,
            block_info: reader.block_info,
            _tiling_order: PhantomData::<T>.runtime(),
        }
    }
}

#[cube]
impl<E: Numeric, T: TilingOrder> Loader<Line<E>> for RhsTensorLoader<E, T> {
    type Gmem = Tensor<Line<E>>;
    type TileReader = RhsSmemTileReader<E, T>;

    fn new(
        gmem: Tensor<Line<E>>,
        gmem_layout: MatrixLayout,
        #[comptime] block_info: BlockInfo,
    ) -> RhsTensorLoader<E, T> {
        let line_size = gmem.line_size();
        let smem = SharedMemory::new_lined(
            comptime!(total_num_elements(block_info) / line_size),
            line_size,
        );

        RhsTensorLoader::<E, T> {
            gmem,
            smem,
            gmem_layout,
            cube_offset: CUBE_POS_Y,
            block_info: block_info.runtime(),
            _tiling_order: PhantomData::<T>.runtime(),
        }
    }

    fn load_block(reader: &mut Self, k_offset: u32) -> Self::TileReader {
        // Assuming RowMajor layout
        // TODO refactor
        let is_row_major = true;
        let (gmem_row_offset, gmem_col_offset) = match is_row_major {
            true => (k_offset, reader.cube_offset),
            false => (reader.cube_offset, k_offset),
        };

        Tensor2SmemContinuous::tensor_to_shared_memory(
            &reader.gmem,
            &mut reader.smem,
            gmem_row_offset,
            gmem_col_offset,
            reader.block_info,
        );

        RhsSmemTileReader::<E, T> {
            smem: reader.smem,
            block_info: reader.block_info,
            _tiling_order: PhantomData::<T>.runtime(),
        }
    }
}
