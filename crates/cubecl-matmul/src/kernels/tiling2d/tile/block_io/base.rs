use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::kernels::tiling2d::config::CubeTiling2dConfig;
use crate::kernels::tiling2d::tile::loader::{CheckBounds, ReadTileInfo};
use crate::kernels::tiling2d::tile::memory_access::ContiguousAccess;
use crate::kernels::tiling2d::write_output::WriteTileInfo;

#[cube]
pub(crate) trait BlockLoader<N: Numeric>: Send + Sync + 'static {
    fn load_tile_plain<A: ContiguousAccess<N>>(
        tensor: &Tensor<Line<N>>,
        shared_memory: &mut SharedMemory<Line<N>>,
        read_tile_info: ReadTileInfo,
        #[comptime] config: CubeTiling2dConfig,
        check_bounds: CheckBounds,
    );

    fn load_tile_transposed(
        tensor: &Tensor<Line<N>>,
        shared_memory: &mut SharedMemory<Line<N>>,
        read_tile_info: ReadTileInfo,
        #[comptime] config: CubeTiling2dConfig,
        check_bounds: CheckBounds,
    );
}

#[cube]
pub(crate) trait BlockWriter<N: Numeric>: Send + Sync + 'static {
    fn write_output<A: ContiguousAccess<N>>(
        out: &mut Tensor<Line<N>>,
        results: &Array<N>,
        write_tile_info: WriteTileInfo,
        #[comptime] config: CubeTiling2dConfig,
        check_bounds: CheckBounds,
    );
}

#[cube]
pub(crate) fn all_zeros_runtime<N: Numeric>(
    shared_memory: &mut SharedMemory<Line<N>>,
    start: u32,
    sm_position_base: u32,
    sm_stride: u32,
    #[comptime] config: CubeTiling2dConfig,
) {
    let tile_size = config.tile_size;
    let zeros = Line::empty(tile_size).fill(N::from_int(0));

    for i in start..tile_size {
        let sm_position = (sm_position_base + i * sm_stride) / tile_size;

        shared_memory[sm_position] = zeros;
    }
}

#[cube]
pub(crate) fn all_zeros_comptime<N: Numeric>(
    shared_memory: &mut SharedMemory<Line<N>>,
    sm_position_base: u32,
    sm_stride: u32,
    #[comptime] config: CubeTiling2dConfig,
) {
    let tile_size = config.tile_size;
    let unroll = config.unroll_tile;
    let zeros = Line::empty(tile_size).fill(N::from_int(0));

    #[unroll(unroll)]
    for i in 0..tile_size {
        let sm_position = (sm_position_base + i * sm_stride) / tile_size;

        shared_memory[sm_position] = zeros;
    }
}
