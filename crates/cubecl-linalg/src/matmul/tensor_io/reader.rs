use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma_matmul::BlockInfo;
use crate::matmul::cube_matmul::smem::fill_shared_memory;
use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::tile_io::reader::{SmemLhsReader, SmemRhsReader};

use super::TensorLoader;

#[derive(CubeType)]
pub struct LhsTensorReader<E: Numeric> {
    smem: SharedMemory<Line<E>>,
    // TODO maybe shouldn't be owned, should have &'a
    gmem: Tensor<Line<E>>,
    gmem_layout: MatrixLayout,
    cube_offset: u32,
    block_info: BlockInfo,
}

#[derive(CubeType)]
pub struct RhsTensorReader<E: Numeric> {
    smem: SharedMemory<Line<E>>,
    // TODO maybe shouldn't be owned, should have &'a
    gmem: Tensor<Line<E>>,
    gmem_layout: MatrixLayout,
    cube_offset: u32,
    block_info: BlockInfo,
}
#[cube]
impl<E: Numeric> TensorLoader<E> for LhsTensorReader<E> {
    type TileReader = SmemLhsReader<E>;

    fn load_tile(reader: &mut Self, k_offset: u32) -> Self::TileReader {
        // Assuming RowMajor layout
        let gmem_x_offset = reader.cube_offset;
        let gmem_y_offset = k_offset;

        fill_shared_memory(
            &reader.gmem,
            &mut reader.smem,
            gmem_x_offset,
            gmem_y_offset,
            reader.block_info,
        );

        SmemLhsReader::<E> {
            memory: reader.smem,
            block_info: reader.block_info,
        }
    }
}
#[cube]
impl<E: Numeric> TensorLoader<E> for RhsTensorReader<E> {
    type TileReader = SmemRhsReader<E>;

    fn load_tile(reader: &mut Self, k_offset: u32) -> Self::TileReader {
        // Assuming RowMajor layout
        let gmem_x_offset = k_offset;
        let gmem_y_offset = reader.cube_offset;

        fill_shared_memory(
            &reader.gmem,
            &mut reader.smem,
            gmem_x_offset,
            gmem_y_offset,
            reader.block_info,
        );

        SmemRhsReader::<E> {
            memory: reader.smem,
            block_info: reader.block_info,
        }
    }
}
