use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma::base::RuntimeCmmaInfo;

use super::super::{
    block_io::{
        base::BlockWriter, horizontal_block_check::HorizontalCheckBlockIO,
        unchecked_block::UncheckedBlockIO, vertical_block_check::VerticalCheckBlockIO,
        whole_block_check::WholeCheckBlockIO,
    },
    config::ComptimeCmmaInfo,
};

#[cube]
/// Writes accumulators to global memory
pub(crate) trait OutputWriter: Send + Sync + 'static {
    fn write_to_output<F: Float>(
        out: &mut Tensor<F>,
        accumulators: Sequence<cmma::Matrix<F>>,
        runtime_info: RuntimeCmmaInfo,
        comptime_info: Comptime<ComptimeCmmaInfo>,
    );
}

#[cube]
pub(crate) fn shared_memory_to_output<F: Float>(
    out: &mut Tensor<F>,
    smem_position: UInt,
    accumulator_sm: SharedMemory<F>,
    n_iter: UInt,
    runtime_info: RuntimeCmmaInfo,
    comptime_info: Comptime<ComptimeCmmaInfo>,
) {
    let check_m_bounds = Comptime::map(comptime_info, |c| c.check_m_bounds);
    let check_n_bounds = Comptime::map(comptime_info, |c| c.check_n_bounds);

    if Comptime::get(check_m_bounds) {
        if Comptime::get(check_n_bounds) {
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
    } else if Comptime::get(check_n_bounds) {
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
    smem_position: UInt,
    accumulator_sm: SharedMemory<F>,
    n_iter: UInt,
    runtime_info: RuntimeCmmaInfo,
    comptime_info: Comptime<ComptimeCmmaInfo>,
) {
    let tile_size = Comptime::map(comptime_info, |c| c.tile_size);
    let tile_size_r = Comptime::runtime(tile_size);
    let num_accumulators = Comptime::map(comptime_info, |c| c.num_accumulators);
    let block_size_n = Comptime::map(comptime_info, |c| c.block_size_n);
    let num_accum_groups_in_block_row =
        Comptime::runtime(block_size_n / (tile_size * num_accumulators));

    let out_vec = Comptime::vectorization(out);
    let out_vec_r = Comptime::runtime(out_vec);
    let n_units_per_tile_row = Comptime::runtime(tile_size / out_vec);
    let sm_stride = Comptime::runtime(tile_size * tile_size);
    let coop_dim = Comptime::map(comptime_info, |c| c.num_coops);

    let dims = runtime_info.dims;
    let ids = runtime_info.ids;
    let offsets = runtime_info.offsets;

    let tile_row = ids.coop / num_accum_groups_in_block_row;
    let tile_col = (ids.coop % num_accum_groups_in_block_row) * num_accum_groups_in_block_row;

    let num_unit_writes = tile_size * tile_size / (out_vec * coop_dim);

    let smem_offset = smem_position * sm_stride + ids.lane * out_vec_r;
    let sm_step = Comptime::runtime(coop_dim * out_vec);

    let lane_row_step = Comptime::runtime(coop_dim * out_vec / tile_size);
    let unit_write_row = ids.lane / n_units_per_tile_row;
    let unit_write_col = ids.lane % n_units_per_tile_row * out_vec_r;

    let row_offset = offsets.cube_row + tile_row * tile_size_r;
    let write_col =
        offsets.cube_col + tile_col * tile_size_r + unit_write_col + n_iter * tile_size_r;

    for i in range(0u32, Comptime::get(num_unit_writes), Comptime::new(true)) {
        let read_pos = smem_offset + i * sm_step;
        let write_row = row_offset + unit_write_row + i * lane_row_step;

        W::write_output(
            out,
            accumulator_sm,
            offsets.batch_out,
            read_pos,
            write_row,
            write_col,
            dims,
        );
    }
}
