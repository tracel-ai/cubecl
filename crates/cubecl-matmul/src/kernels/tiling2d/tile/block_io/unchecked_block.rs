use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::kernels::tiling2d::{
    config::CubeTiling2dConfig,
    tile::{
        loader::{CheckBounds, ReadTileInfo},
        memory_access::{ContiguousAccess, StridedAccess, UnmatchingVectorization, WritePositions},
    },
    write_output::WriteTileInfo,
};

use super::base::{BlockLoader, BlockWriter};

/// Assumes block sizes divide tensor shape
pub(crate) struct UncheckedBlockIO;

#[cube]
impl<N: Numeric> BlockLoader<N> for UncheckedBlockIO {
    fn load_tile_plain<A: ContiguousAccess<N>>(
        tensor: &Tensor<Line<N>>,
        shared_memory: &mut SharedMemory<Line<N>>,
        info: ReadTileInfo,
        #[comptime] config: CubeTiling2dConfig,
        _check_bounds: CheckBounds,
    ) {
        let tile_size = config.tile_size;
        let unroll = config.unroll_tile;
        let line_size = tensor.line_size().runtime();

        #[unroll(unroll)]
        for i in 0..tile_size {
            let gm_position = (info.gm_position_base + i * info.gm_stride) / line_size;
            let sm_position = (info.sm_position_base + i * info.sm_stride) / tile_size;

            shared_memory[sm_position] = A::read_contiguous_unchecked(tensor, gm_position, config);
        }
    }

    fn load_tile_transposed(
        tensor: &Tensor<Line<N>>,
        shared_memory: &mut SharedMemory<Line<N>>,
        info: ReadTileInfo,
        #[comptime] config: CubeTiling2dConfig,
        _check_bounds: CheckBounds,
    ) {
        let tile_size = config.tile_size;
        let unroll = config.unroll_tile;

        #[unroll(unroll)]
        for i in 0..tile_size {
            let gm_position = info.gm_position_base + i;
            let sm_position = (info.sm_position_base + i * info.sm_stride) / tile_size;

            shared_memory[sm_position] = UnmatchingVectorization::read_strided_unchecked(
                tensor,
                gm_position,
                info.gm_stride,
                config,
            );
        }
    }
}

#[cube]
impl<N: Numeric> BlockWriter<N> for UncheckedBlockIO {
    fn write_output<A: ContiguousAccess<N>>(
        out: &mut Tensor<Line<N>>,
        results: &Array<N>,
        info: WriteTileInfo,
        #[comptime] config: CubeTiling2dConfig,
        _check_bounds: CheckBounds,
    ) {
        let tile_size = config.tile_size;
        let unroll = config.unroll_tile;
        let coordinates = info.coordinates;

        let row = coordinates.skip_row + coordinates.unit_row;
        let col = coordinates.skip_col + coordinates.unit_col;
        let out_position_base = row * info.out_stride + col + info.offset_output;

        #[unroll(unroll)]
        for result_index in 0..tile_size {
            let positions = WritePositions {
                result: result_index * tile_size,
                out: out_position_base + result_index * info.out_stride,
            };

            A::write_contiguous_unchecked(out, results, positions, config);
        }
    }
}
