use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::block_info::{BlockInfo, total_num_elements};
use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::tile_io::loading::Tensor2Smem;
use crate::matmul::tile_io::loading::Tensor2SmemContinuous;
use crate::matmul::tile_io::loading::{LhsSmemTileReader, RhsSmemTileReader};
use crate::matmul::tile_io::Loader;

#[derive(CubeType)]
pub struct LhsTensorLoader<E: Numeric> {
    pub gmem: Tensor<Line<E>>,
    pub smem: SharedMemory<Line<E>>,
    pub gmem_layout: MatrixLayout,
    pub cube_offset: u32,
    pub block_info: BlockInfo,
}

#[derive(CubeType)]
pub struct RhsTensorLoader<E: Numeric> {
    pub gmem: Tensor<Line<E>>,
    pub smem: SharedMemory<Line<E>>,
    pub gmem_layout: MatrixLayout,
    pub cube_offset: u32,
    pub block_info: BlockInfo,
}

#[cube]
pub(crate) fn new_lhs_tensor_loader<E: Numeric>(
    gmem: Tensor<Line<E>>,
    gmem_layout: MatrixLayout,
    #[comptime] block_info: BlockInfo,
) -> LhsTensorLoader<E> {
    let line_size = gmem.line_size();
    let smem = SharedMemory::new_lined(
        comptime!(total_num_elements(block_info) / line_size),
        line_size,
    );

    LhsTensorLoader::<E> {
        gmem,
        smem,
        gmem_layout,
        cube_offset: CUBE_POS_X,
        block_info: block_info.runtime(),
    }
}

#[cube]
pub(crate) fn new_rhs_tensor_loader<E: Numeric>(
    gmem: Tensor<Line<E>>,
    gmem_layout: MatrixLayout,
    #[comptime] block_info: BlockInfo,
) -> RhsTensorLoader<E> {
    let line_size = gmem.line_size();
    let smem = SharedMemory::new_lined(
        comptime!(total_num_elements(block_info) / line_size),
        line_size,
    );

    RhsTensorLoader::<E> {
        gmem,
        smem,
        gmem_layout,
        cube_offset: CUBE_POS_Y,
        block_info: block_info.runtime(),
    }
}

#[cube]
impl<E: Numeric> Loader<Line<E>> for LhsTensorLoader<E> {
    type TileReader = LhsSmemTileReader<E>;

    fn load_block(reader: &mut Self, k_offset: u32) -> Self::TileReader {
        // Assuming RowMajor layout
        let gmem_row_offset = reader.cube_offset;
        let gmem_col_offset = k_offset;

        Tensor2SmemContinuous::tensor_to_shared_memory(
            &reader.gmem,
            &mut reader.smem,
            gmem_row_offset,
            gmem_col_offset,
            reader.block_info,
        );

        LhsSmemTileReader::<E> {
            smem: reader.smem,
            block_info: reader.block_info,
        }
    }
}

#[cube]
impl<E: Numeric> Loader<Line<E>> for RhsTensorLoader<E> {
    type TileReader = RhsSmemTileReader<E>;

    fn load_block(reader: &mut Self, k_offset: u32) -> Self::TileReader {
        // Assuming RowMajor layout
        let gmem_row_offset = k_offset;
        let gmem_col_offset = reader.cube_offset;

        Tensor2SmemContinuous::tensor_to_shared_memory(
            &reader.gmem,
            &mut reader.smem,
            gmem_row_offset,
            gmem_col_offset,
            reader.block_info,
        );

        RhsSmemTileReader::<E> {
            smem: reader.smem,
            block_info: reader.block_info,
        }
    }
}
