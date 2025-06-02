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

use super::base::{BlockLoader, BlockWriter, all_zeros_comptime, all_zeros_runtime};

pub(crate) struct HorizontalCheckBlockIO;

#[cube]
impl<N: Numeric> BlockLoader<N> for HorizontalCheckBlockIO {
    fn load_tile_plain<A: ContiguousAccess<N>>(
        tensor: &Tensor<Line<N>>,
        shared_memory: &mut SharedMemory<Line<N>>,
        info: ReadTileInfo,
        #[comptime] config: CubeTiling2dConfig,
        check_bounds: CheckBounds,
    ) {
        let tile_size = config.tile_size;
        let line_size = tensor.line_size();
        let unroll = config.unroll_tile;

        let col = check_bounds.skip_col + info.read_col;
        if check_bounds.dim_horizontal > col {
            #[unroll(unroll)]
            for i in 0..tile_size {
                let gm_position = (info.gm_position_base + i * info.gm_stride) / line_size;
                let sm_position = (info.sm_position_base + i * info.sm_stride) / tile_size;

                shared_memory[sm_position] =
                    A::read_contiguous_checked(tensor, gm_position, check_bounds, info, config);
            }
        } else {
            all_zeros_comptime(shared_memory, info.sm_position_base, info.sm_stride, config);
        }
    }

    fn load_tile_transposed(
        tensor: &Tensor<Line<N>>,
        shared_memory: &mut SharedMemory<Line<N>>,
        info: ReadTileInfo,
        #[comptime] config: CubeTiling2dConfig,
        check_bounds: CheckBounds,
    ) {
        let tile_size = config.tile_size;

        let mut num_reads = 0;
        let col = check_bounds.skip_col + info.read_col;
        let dim_horizontal = check_bounds.dim_horizontal;
        if dim_horizontal > col {
            num_reads = Min::min(dim_horizontal - col, tile_size);
        }

        for i in 0..num_reads {
            let gm_position = info.gm_position_base + i;
            let sm_position = (info.sm_position_base + i * info.sm_stride) / tile_size;

            shared_memory[sm_position] = UnmatchingVectorization::read_strided_unchecked(
                tensor,
                gm_position,
                info.gm_stride,
                config,
            );
        }

        all_zeros_runtime(
            shared_memory,
            num_reads,
            info.sm_position_base,
            info.sm_stride,
            config,
        );
    }
}

#[cube]
impl<N: Numeric> BlockWriter<N> for HorizontalCheckBlockIO {
    fn write_output<A: ContiguousAccess<N>>(
        out: &mut Tensor<Line<N>>,
        results: &Array<N>,
        info: WriteTileInfo,
        #[comptime] config: CubeTiling2dConfig,
        check_bounds: CheckBounds,
    ) {
        let tile_size = config.tile_size;
        let unroll = config.unroll_tile;
        let coordinates = info.coordinates;

        let col = coordinates.skip_col + coordinates.unit_col;

        if check_bounds.dim_horizontal > col {
            let row = coordinates.skip_row + coordinates.unit_row;
            let out_position_base = row * info.out_stride + col + info.offset_output;

            #[unroll(unroll)]
            for result_index in 0..tile_size {
                let positions = WritePositions {
                    result: result_index * tile_size,
                    out: out_position_base + result_index * info.out_stride,
                };

                A::write_contiguous_checked(out, results, positions, check_bounds, col, config);
            }
        }
    }
}
