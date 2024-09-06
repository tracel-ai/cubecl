use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma::base::Ids;

use super::super::{
    base::{Dimensions, Offsets},
    block_io::{
        base::BlockWriter, horizontal_block_check::HorizontalCheckBlockIO,
        unchecked_block::UncheckedBlockIO, vertical_block_check::VerticalCheckBlockIO,
        whole_block_check::WholeCheckBlockIO,
    },
    config::CmmaComptimeInfo,
};

#[cube]
/// Writes accumulators to global memory
pub(crate) trait OutputWriter: Send + Sync + 'static {
    fn write_to_output<F: Float>(
        out: &mut Tensor<F>,
        accumulators: Sequence<cmma::Matrix<F>>,
        offsets: Offsets,
        dims: Dimensions,
        config: Comptime<CmmaComptimeInfo>,
        ids: Ids,
    );
}

#[cube]
pub(crate) fn shared_memory_to_output<F: Float>(
    out: &mut Tensor<F>,
    offsets: Offsets,
    smem_position: UInt,
    accumulator_sm: SharedMemory<F>,
    dims: Dimensions,
    config: Comptime<CmmaComptimeInfo>,
    n_iter: UInt,
    ids: Ids,
) {
    let check_m_bounds = Comptime::map(config, |c| c.check_m_bounds);
    let check_n_bounds = Comptime::map(config, |c| c.check_n_bounds);

    if Comptime::get(check_m_bounds) {
        if Comptime::get(check_n_bounds) {
            write_tile::<F, WholeCheckBlockIO>(
                out,
                offsets,
                smem_position,
                accumulator_sm,
                dims,
                config,
                n_iter,
                ids,
            );
        } else {
            write_tile::<F, VerticalCheckBlockIO>(
                out,
                offsets,
                smem_position,
                accumulator_sm,
                dims,
                config,
                n_iter,
                ids,
            );
        }
    } else if Comptime::get(check_n_bounds) {
        write_tile::<F, HorizontalCheckBlockIO>(
            out,
            offsets,
            smem_position,
            accumulator_sm,
            dims,
            config,
            n_iter,
            ids,
        );
    } else {
        write_tile::<F, UncheckedBlockIO>(
            out,
            offsets,
            smem_position,
            accumulator_sm,
            dims,
            config,
            n_iter,
            ids,
        );
    }
}

#[cube]
fn write_tile<F: Float, W: BlockWriter<F>>(
    out: &mut Tensor<F>,
    offsets: Offsets,
    smem_position: UInt,
    accumulator_sm: SharedMemory<F>,
    dims: Dimensions,
    config: Comptime<CmmaComptimeInfo>,
    n_iter: UInt,
    ids: Ids,
) {
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let tile_size_r = Comptime::runtime(tile_size);
    let num_accumulators = Comptime::map(config, |c| c.num_accumulators);
    let block_size_n = Comptime::map(config, |c| c.block_size_n);
    let num_accum_groups_in_block_row =
        Comptime::runtime(block_size_n / (tile_size * num_accumulators));

    let out_vec = Comptime::vectorization(out);
    let out_vec_r = Comptime::runtime(out_vec);
    let n_units_per_tile_row = Comptime::runtime(tile_size / out_vec);
    let sm_stride = Comptime::runtime(tile_size * tile_size);
    let coop_dim = Comptime::map(config, |c| c.coop_dim);

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
