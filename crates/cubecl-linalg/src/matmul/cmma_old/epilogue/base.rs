use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::super::prologue::RuntimeCmmaInfo;

use super::super::{
    block_io::{
        base::BlockWriter, horizontal_block_check::HorizontalCheckBlockIO,
        unchecked_block::UncheckedBlockIO, vertical_block_check::VerticalCheckBlockIO,
        whole_block_check::WholeCheckBlockIO,
    },
    config::{ComptimeCmmaInfo, WriteOutStrategy},
};
use super::{large_smem::LargeSmemWriter, reuse_smem::ReuseSmemWriter};

#[cube]
pub(crate) fn write_to_output<F: Float>(
    out: &mut Tensor<F>,
    accumulators: Sequence<cmma::Matrix<F>>,
    runtime_info: RuntimeCmmaInfo,
    #[comptime] comptime_info: ComptimeCmmaInfo,
) {
    match comptime_info.write_out_strategy {
        WriteOutStrategy::LargeSmem => {
            LargeSmemWriter::write_to_output(out, accumulators, runtime_info, comptime_info);
        }
        WriteOutStrategy::ReuseSmem => {
            ReuseSmemWriter::write_to_output(out, accumulators, runtime_info, comptime_info);
        }
    }
}

#[cube]
/// Writes accumulators to global memory
pub(crate) trait OutputWriter: Send + Sync + 'static {
    fn write_to_output<F: Float>(
        out: &mut Tensor<F>,
        accumulators: Sequence<cmma::Matrix<F>>,
        runtime_info: RuntimeCmmaInfo,
        #[comptime] comptime_info: ComptimeCmmaInfo,
    );
}

#[cube]
pub(crate) fn shared_memory_to_output<F: Float>(
    out: &mut Tensor<F>,
    smem_position: u32,
    accumulator_sm: SharedMemory<F>,
    n_iter: u32,
    runtime_info: RuntimeCmmaInfo,
    #[comptime] comptime_info: ComptimeCmmaInfo,
) {
    let check_m_bounds = comptime_info.check_m_bounds;
    let check_n_bounds = comptime_info.check_n_bounds;

    if check_m_bounds {
        if check_n_bounds {
            write_tile::<F, WholeCheckBlockIO>(
                out,
                smem_position,
                accumulator_sm,
                n_iter,
                runtime_info,
                comptime_info,
            );
        } else {
            write_tile::<F, VerticalCheckBlockIO>(
                out,
                smem_position,
                accumulator_sm,
                n_iter,
                runtime_info,
                comptime_info,
            );
        }
    } else if check_n_bounds {
        write_tile::<F, HorizontalCheckBlockIO>(
            out,
            smem_position,
            accumulator_sm,
            n_iter,
            runtime_info,
            comptime_info,
        );
    } else {
        write_tile::<F, UncheckedBlockIO>(
            out,
            smem_position,
            accumulator_sm,
            n_iter,
            runtime_info,
            comptime_info,
        );
    }
}

#[cube]
fn write_tile<F: Float, W: BlockWriter<F>>(
    out: &mut Tensor<F>,
    smem_position: u32,
    accumulator_sm: SharedMemory<F>,
    n_iter: u32,
    runtime_info: RuntimeCmmaInfo,
    #[comptime] comptime_info: ComptimeCmmaInfo,
) {
    let num_accumulators = comptime_info.num_accumulators;
    let block_size_n = comptime_info.block_size_n;

    let tile_height = comptime_info.tile_size_m;
    let tile_width = comptime_info.tile_size_n;
    let num_tile_elements = tile_height * tile_width;

    let num_accum_groups_in_block_row = block_size_n / (tile_width * num_accumulators);

    let out_vec = vectorization_of(out);
    let n_units_per_tile_row = tile_height / out_vec;
    let smem_stride = num_tile_elements;
    let plane_dim = comptime_info.plane_dim;

    let plane_id = runtime_info.compute_ids.plane;
    let lane_id = runtime_info.compute_ids.lane;

    let offsets = runtime_info.offsets;

    let tile_row = plane_id / num_accum_groups_in_block_row;
    let tile_col = (plane_id % num_accum_groups_in_block_row) * num_accumulators;

    let num_unit_writes = num_tile_elements / (out_vec * plane_dim);

    let smem_offset = smem_position * smem_stride + lane_id * out_vec;
    let smem_step = plane_dim * out_vec;

    let lane_row_step = plane_dim * out_vec / tile_height;
    let unit_write_row = lane_id / n_units_per_tile_row;
    let unit_write_col = lane_id % n_units_per_tile_row * out_vec;

    let row_offset = offsets.cube_row + tile_row * tile_height;
    let write_col = offsets.cube_col + tile_col * tile_width + unit_write_col + n_iter * tile_width;

    #[unroll]
    for i in 0..num_unit_writes {
        let read_pos = smem_offset + i * smem_step;
        let write_row = row_offset + unit_write_row + i * lane_row_step;

        W::write_single(
            out,
            accumulator_sm,
            offsets.batch_out,
            read_pos,
            write_row,
            write_col,
            runtime_info.dims,
        );
    }
}
