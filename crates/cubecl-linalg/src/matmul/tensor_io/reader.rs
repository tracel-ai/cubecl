use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma_matmul::num_elements;
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

#[cube]
pub(crate) fn new_lhs_tensor_reader<E: Numeric>(
    gmem: Tensor<Line<E>>,
    gmem_layout: MatrixLayout,
    block_info: BlockInfo,
) -> LhsTensorReader<E> {
    let line_size = gmem.line_size();
    let smem = SharedMemory::new_lined(num_elements(&block_info) / line_size, line_size);

    LhsTensorReader::<E> {
        smem,
        gmem,
        gmem_layout,
        cube_offset: CUBE_POS_X,
        block_info,
    }
}

#[cube]
pub(crate) fn new_rhs_tensor_reader<E: Numeric>(
    gmem: Tensor<Line<E>>,
    gmem_layout: MatrixLayout,
    block_info: BlockInfo,
) -> RhsTensorReader<E> {
    let line_size = gmem.line_size();
    let smem = SharedMemory::new_lined(num_elements(&block_info) / line_size, line_size);

    RhsTensorReader::<E> {
        smem,
        gmem,
        gmem_layout,
        cube_offset: CUBE_POS_X,
        block_info,
    }
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

impl<E: Numeric> RhsTensorReader<E> {
    pub(crate) fn new(
        gmem: Tensor<Line<E>>,
        gmem_layout: MatrixLayout,
        block_info: BlockInfo,
    ) -> Self {
        let line_size = gmem.line_size();
        let smem = SharedMemory::new_lined(num_elements(&block_info) / line_size, line_size);

        Self {
            smem,
            gmem,
            gmem_layout,
            cube_offset: CUBE_POS_Y,
            block_info,
        }
    }
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
