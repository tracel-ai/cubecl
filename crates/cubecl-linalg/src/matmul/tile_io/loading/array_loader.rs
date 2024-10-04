use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma_matmul::BlockInfo;
use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::tile_io::loading::{array_to_shared_memory, RhsSmemTileReader};
use crate::matmul::tile_io::Loader;

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
    type TileReader = LhsSmemTileReader<E>;

    fn load_block(reader: &mut Self, _k_offset: u32) -> Self::TileReader {
        array_to_shared_memory(&reader.gmem, &mut reader.smem, reader.block_info);

        LhsSmemTileReader::<E> {
            smem: reader.smem,
            block_info: reader.block_info,
        }
    }
}

#[cube]
impl<E: Numeric> Loader<Line<E>> for RhsArrayLoader<E> {
    type TileReader = RhsSmemTileReader<E>;

    // TODO bad api if k_offset not useful for all loaders
    fn load_block(reader: &mut Self, _k_offset: u32) -> Self::TileReader {
        array_to_shared_memory(&reader.gmem, &mut reader.smem, reader.block_info);

        RhsSmemTileReader::<E> {
            smem: reader.smem,
            block_info: reader.block_info,
        }
    }
}
