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

use super::base::{BlockLoader, BlockWriter, all_zeros_runtime};

pub(crate) struct VerticalCheckBlockIO;

#[cube]
impl<N: Numeric> BlockLoader<N> for VerticalCheckBlockIO {
    fn load_tile_plain<A: ContiguousAccess<N>>(
        tensor: &Tensor<Line<N>>,
        shared_memory: &mut SharedMemory<Line<N>>,
        info: ReadTileInfo,
        #[comptime] config: CubeTiling2dConfig,
        check_bounds: CheckBounds,
    ) {
        let tile_size = config.tile_size;
        let line_size = tensor.line_size().runtime();

        let mut num_reads = 0;
        let row = check_bounds.skip_row + info.read_row;
        if check_bounds.dim_vertical > row {
            num_reads = Min::min(check_bounds.dim_horizontal - row, tile_size);
        }

        for i in 0..num_reads {
            let gm_position = (info.gm_position_base + i * info.gm_stride) / line_size;
            let sm_position = (info.sm_position_base + i * info.sm_stride) / tile_size;

            shared_memory[sm_position] = A::read_contiguous_unchecked(tensor, gm_position, config);
        }

        all_zeros_runtime(
            shared_memory,
            num_reads,
            info.sm_position_base,
            info.sm_stride,
            config,
        );
    }

    fn load_tile_transposed(
        tensor: &Tensor<Line<N>>,
        shared_memory: &mut SharedMemory<Line<N>>,
        info: ReadTileInfo,
        #[comptime] config: CubeTiling2dConfig,
        check_bounds: CheckBounds,
    ) {
        let tile_size = config.tile_size;
        let unroll = config.unroll_tile;

        #[unroll(unroll)]
        for i in 0..tile_size {
            let gm_position = info.gm_position_base + i;
            let sm_position = (info.sm_position_base + i * info.sm_stride) / tile_size;

            shared_memory[sm_position] = UnmatchingVectorization::read_strided_checked(
                tensor,
                gm_position,
                info.gm_stride,
                check_bounds,
                info,
                config,
            );
        }
    }
}

#[cube]
impl<N: Numeric> BlockWriter<N> for VerticalCheckBlockIO {
    fn write_output<A: ContiguousAccess<N>>(
        out: &mut Tensor<Line<N>>,
        results: &Array<N>,
        info: WriteTileInfo,
        #[comptime] config: CubeTiling2dConfig,
        check_bounds: CheckBounds,
    ) {
        let tile_size = config.tile_size;
        let coordinates = info.coordinates;

        let row = coordinates.skip_row + coordinates.unit_row;
        let col = coordinates.skip_col + coordinates.unit_col;
        let out_position_base = row * info.out_stride + col + info.offset_output;

        let mut num_writes = 0;
        if check_bounds.dim_vertical > row {
            num_writes = Min::min(check_bounds.dim_vertical - row, tile_size);
        }

        for result_index in 0..num_writes {
            let positions = WritePositions {
                result: result_index * tile_size,
                out: out_position_base + result_index * info.out_stride,
            };

            A::write_contiguous_unchecked(out, results, positions, config);
        }
    }
}
