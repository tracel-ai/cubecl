use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma::config::CmmaConfig;

#[derive(CubeType, Copy, Clone)]
pub(crate) struct CheckBounds {
    pub dim_vertical: UInt,
    pub dim_horizontal: UInt,
    pub skip_row: UInt,
    pub skip_col: UInt,
}

#[derive(CubeType, Copy, Clone)]
pub(crate) struct ReadTileInfo {
    // pub read_row: UInt,
    // pub read_col: UInt,
    // pub gm_position_base: UInt,
    // pub sm_position_base: UInt,
    // pub gm_stride: UInt,
    // pub sm_stride: UInt,
}

#[derive(CubeType)]
pub(crate) struct WriteTileInfo {
    // pub coordinates: Coordinates,
    // pub offset_output: UInt,
    // pub out_stride: UInt,
}

#[cube]
pub(crate) trait BlockLoader<F: Float, FC: Float>: Send + Sync + 'static {
    fn load_tile(
        tensor: &Tensor<F>,
        shared_memory: &mut SharedMemory<FC>,
        batch_offset: UInt,
        read_row: UInt,
        read_col: UInt,
        write_pos: UInt,
        dim_vertical: UInt,
        dim_horizontal: UInt,
    );
}

#[cube]
pub(crate) trait BlockWriter<F: Float>: Send + Sync + 'static {
    fn write_output(
        out: &mut Tensor<F>,
        results: &Array<F>,
        write_tile_info: WriteTileInfo,
        config: Comptime<CmmaConfig>,
        check_bounds: CheckBounds,
    );
}
