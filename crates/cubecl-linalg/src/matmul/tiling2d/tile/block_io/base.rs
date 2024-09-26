use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::tiling2d::config::CubeTiling2dConfig;
use crate::matmul::tiling2d::tile::loader::{CheckBounds, ReadTileInfo};
use crate::matmul::tiling2d::tile::memory_access::ContiguousAccess;
use crate::matmul::tiling2d::write_output::WriteTileInfo;

#[cube]
pub(crate) trait BlockLoader<F: Float>: Send + Sync + 'static {
    fn load_tile_plain<A: ContiguousAccess<F>>(
        tensor: &Tensor<Line<F>>,
        shared_memory: &mut SharedMemory<Line<F>>,
        read_tile_info: ReadTileInfo,
        #[comptime] config: CubeTiling2dConfig,
        check_bounds: CheckBounds,
    );

    fn load_tile_transposed(
        tensor: &Tensor<Line<F>>,
        shared_memory: &mut SharedMemory<Line<F>>,
        read_tile_info: ReadTileInfo,
        #[comptime] config: CubeTiling2dConfig,
        check_bounds: CheckBounds,
    );
}

#[cube]
pub(crate) trait BlockWriter<F: Float>: Send + Sync + 'static {
    fn write_output<A: ContiguousAccess<F>>(
        out: &mut Tensor<Line<F>>,
        results: &Array<F>,
        write_tile_info: WriteTileInfo,
        #[comptime] config: CubeTiling2dConfig,
        check_bounds: CheckBounds,
    );
}

#[cube]
pub(crate) fn all_zeros_runtime<F: Float>(
    shared_memory: &mut SharedMemory<Line<F>>,
    start: u32,
    sm_position_base: u32,
    sm_stride: u32,
    #[comptime] config: CubeTiling2dConfig,
) {
    let tile_size = config.tile_size;
    let zeros = Line::empty(tile_size).fill(F::new(0.));

    for i in start..tile_size {
        let sm_position = (sm_position_base + i * sm_stride) / tile_size;

        shared_memory[sm_position] = zeros;
    }
}

#[cube]
pub(crate) fn all_zeros_comptime<F: Float>(
    shared_memory: &mut SharedMemory<Line<F>>,
    sm_position_base: u32,
    sm_stride: u32,
    #[comptime] config: CubeTiling2dConfig,
) {
    let tile_size = config.tile_size;
    let unroll = config.unroll_tile;
    let zeros = Line::empty(tile_size).fill(F::new(0.));

    #[unroll(unroll)]
    for i in 0..tile_size {
        let sm_position = (sm_position_base + i * sm_stride) / tile_size;

        shared_memory[sm_position] = zeros;
    }
}
