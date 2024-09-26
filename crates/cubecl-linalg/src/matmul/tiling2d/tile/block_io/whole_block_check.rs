use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::tiling2d::{
    config::CubeTiling2dConfig,
    tile::{
        loader::{CheckBounds, ReadTileInfo},
        memory_access::{ContiguousAccess, StridedAccess, UnmatchingVectorization, WritePositions},
    },
    write_output::WriteTileInfo,
};

use super::base::{all_zeros_comptime, all_zeros_runtime, BlockLoader, BlockWriter};

pub(crate) struct WholeCheckBlockIO;

#[cube]
impl<F: Float> BlockLoader<F> for WholeCheckBlockIO {
    fn load_tile_plain<A: ContiguousAccess<F>>(
        tensor: &Tensor<Line<F>>,
        shared_memory: &mut SharedMemory<Line<F>>,
        info: ReadTileInfo,
        #[comptime] config: CubeTiling2dConfig,
        check_bounds: CheckBounds,
    ) {
        let tile_size = config.tile_size;
        let vectorization = vectorization_of(tensor);

        let col = check_bounds.skip_col + info.read_col;
        if check_bounds.dim_horizontal > col {
            let mut num_reads_vertical = 0;
            let row = check_bounds.skip_row + info.read_row;
            if check_bounds.dim_vertical > row {
                num_reads_vertical = Min::min(check_bounds.dim_vertical - row, tile_size);
            }

            for i in 0..num_reads_vertical {
                let gm_position = (info.gm_position_base + i * info.gm_stride) / vectorization;
                let sm_position = (info.sm_position_base + i * info.sm_stride) / tile_size;

                shared_memory[sm_position] =
                    A::read_contiguous_checked(tensor, gm_position, check_bounds, info, config);
            }

            all_zeros_runtime(
                shared_memory,
                num_reads_vertical,
                info.sm_position_base,
                info.sm_stride,
                config,
            );
        } else {
            all_zeros_comptime(shared_memory, info.sm_position_base, info.sm_stride, config);
        }
    }
    fn load_tile_transposed(
        tensor: &Tensor<Line<F>>,
        shared_memory: &mut SharedMemory<Line<F>>,
        info: ReadTileInfo,
        #[comptime] config: CubeTiling2dConfig,
        check_bounds: CheckBounds,
    ) {
        let tile_size = config.tile_size;

        let mut num_reads_horizontal = 0;
        let col = check_bounds.skip_col + info.read_col;
        let dim_horizontal = check_bounds.dim_horizontal;
        if dim_horizontal > col {
            num_reads_horizontal = Min::min(dim_horizontal - col, tile_size);
        }

        for i in 0..num_reads_horizontal {
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

        all_zeros_runtime(
            shared_memory,
            num_reads_horizontal,
            info.sm_position_base,
            info.sm_stride,
            config,
        );
    }
}

#[cube]
impl<F: Float> BlockWriter<F> for WholeCheckBlockIO {
    fn write_output<A: ContiguousAccess<F>>(
        out: &mut Tensor<Line<F>>,
        results: &Array<F>,
        info: WriteTileInfo,
        #[comptime] config: CubeTiling2dConfig,
        check_bounds: CheckBounds,
    ) {
        let tile_size = config.tile_size;
        let coordinates = info.coordinates;

        let col = coordinates.skip_col + coordinates.unit_col;

        if check_bounds.dim_horizontal > col {
            let mut num_writes_vertical = 0;
            let row = coordinates.skip_row + coordinates.unit_row;

            if check_bounds.dim_vertical > row {
                num_writes_vertical = Min::min(check_bounds.dim_vertical - row, tile_size);
            }

            let out_position_base = row * info.out_stride + col + info.offset_output;

            for result_index in 0..num_writes_vertical {
                let positions = WritePositions {
                    result: result_index * tile_size,
                    out: out_position_base + result_index * info.out_stride,
                };

                A::write_contiguous_checked(out, results, positions, check_bounds, col, config);
            }
        }
    }
}
