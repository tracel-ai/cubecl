use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::base::{Fragments, Ids, SharedMemories};
use super::config::ComptimeCmmaInfo;

#[cube]
#[allow(unused_mut)]
pub(crate) fn compute_loop<F: Float, FC: Float>(
    shared_memories: SharedMemories<FC>,
    fragments: &mut Fragments<F, FC>,
    ids: Ids,
    #[comptime] comptime_info: ComptimeCmmaInfo,
) {
    // Comptime values
    let block_size_k = comptime_info.block_size_k;
    let block_size_n = comptime_info.block_size_n;
    let tile_size = comptime_info.tile_size;
    let unroll = comptime_info.unroll;
    let num_accumulators = comptime_info.num_accumulators;
    let num_buffers = block_size_k / tile_size;
    let smem_stride = tile_size * tile_size;
    let num_coop_per_row = (block_size_n / tile_size) / num_accumulators;

    // Runtime values
    let tile_row = ids.coop / num_coop_per_row;
    let tile_col_base = (ids.coop % num_coop_per_row) * num_accumulators;

    #[unroll(unroll)]
    for buffer_iter in 0..num_buffers {
        #[unroll]
        for accumulator_iter in 0..num_accumulators {
            // Load lhs data into fragment
            let shared_lhs_tile = tile_row * num_buffers + buffer_iter;
            let shared_lhs_pos = shared_lhs_tile * smem_stride;
            let lhs_slice = shared_memories
                .lhs
                .slice(shared_lhs_pos, shared_lhs_pos + smem_stride);
            cmma::load::<FC>(&fragments.lhs, lhs_slice, 16);

            // Load rhs data into fragment
            let tile_col = tile_col_base + accumulator_iter;
            let shared_rhs_tile = tile_col * num_buffers + buffer_iter;
            let shared_rhs_pos = shared_rhs_tile * smem_stride;
            let rhs_slice = shared_memories
                .rhs
                .slice(shared_rhs_pos, shared_rhs_pos + smem_stride);
            cmma::load::<FC>(&fragments.rhs, rhs_slice, 16);

            // Execute cmma and accumulate in accumulator
            let accumulator = &fragments.accumulators.index(accumulator_iter);
            cmma::execute::<FC, FC, F, F>(&fragments.lhs, &fragments.rhs, accumulator, accumulator);
        }
    }
}
