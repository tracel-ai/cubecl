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
    let tile_size = comptime_info.tile_size;
    let num_accumulators = comptime_info.num_accumulators;
    let block_size_n = comptime_info.block_size_n;
    let num_accum_groups_in_block_row = block_size_n / (tile_size * num_accumulators);

    let out_vec = vectorization_of(out);
    let n_units_per_tile_row = tile_size / out_vec;
    let sm_stride = tile_size * tile_size;
    let coop_dim = comptime_info.coop_dim;

    let coop_id = runtime_info.compute_ids.coop;
    let lane_id = runtime_info.compute_ids.lane;

    let offsets = runtime_info.offsets;

    let tile_row = coop_id / num_accum_groups_in_block_row;
    let tile_col = (coop_id % num_accum_groups_in_block_row) * num_accumulators;

    let num_unit_writes = tile_size * tile_size / (out_vec * coop_dim);

    let smem_offset = smem_position * sm_stride + lane_id * out_vec;
    let sm_step = coop_dim * out_vec;

    let lane_row_step = coop_dim * out_vec / tile_size;
    let unit_write_row = lane_id / n_units_per_tile_row;
    let unit_write_col = lane_id % n_units_per_tile_row * out_vec;

    let row_offset = offsets.cube_row + tile_row * tile_size;
    let write_col = offsets.cube_col + tile_col * tile_size + unit_write_col + n_iter * tile_size;

    #[unroll]
    for i in 0..num_unit_writes {
        let read_pos = smem_offset + i * sm_step;
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
