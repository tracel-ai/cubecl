use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::base::{CmmaMatrices, Ids, SharedMemories};
use super::config::ComptimeCmmaInfo;

#[cube]
#[allow(unused_mut)]
pub(crate) fn compute_loop<F: Float, FC: Float>(
    shared_memories: SharedMemories<FC>,
    cmma_matrices: &mut CmmaMatrices<F, FC>,
    ids: Ids,
    comptime_info: Comptime<ComptimeCmmaInfo>,
) {
    let block_size_n = Comptime::map(comptime_info, |c| c.block_size_n);
    let tile_size = Comptime::map(comptime_info, |c| c.tile_size);
    let num_accumulators = Comptime::map(comptime_info, |c| c.num_accumulators);
    let num_coop_per_row = Comptime::runtime((block_size_n / tile_size) / num_accumulators);

    let tile_row = ids.coop / num_coop_per_row;
    let tile_col_base = (ids.coop % num_coop_per_row) * Comptime::runtime(num_accumulators);

    let lhs = &cmma_matrices.lhs;
    let rhs = &cmma_matrices.rhs;
    let accumulators = &cmma_matrices.accumulators;

    for n in range(0u32, Comptime::get(num_accumulators), Comptime::new(true)) {
        compute_tile::<F, FC>(
            tile_row,
            tile_col_base + n,
            shared_memories,
            lhs,
            rhs,
            accumulators.index(n),
            comptime_info,
        );
    }
}

#[cube]
fn compute_tile<F: Float, FC: Float>(
    tile_row: UInt,
    tile_col: UInt,
    shared_memories: SharedMemories<FC>,
    lhs: &cmma::Matrix<FC>,
    rhs: &cmma::Matrix<FC>,
    accumulator: &cmma::Matrix<F>,
    comptime_info: Comptime<ComptimeCmmaInfo>,
) {
    let block_size_k = Comptime::map(comptime_info, |c| c.block_size_k);
    let tile_size = Comptime::map(comptime_info, |c| c.tile_size);
    let unroll = Comptime::map(comptime_info, |c| c.unroll);

    let smem_stride = Comptime::runtime(tile_size * tile_size);
    let num_tiles_in_k = Comptime::runtime(block_size_k / tile_size);

    for k_iter in range(0u32, num_tiles_in_k, unroll) {
        let shared_lhs_tile = tile_row * num_tiles_in_k + k_iter;
        let shared_rhs_tile = tile_col * num_tiles_in_k + k_iter;
        let shared_lhs_pos = shared_lhs_tile * smem_stride;
        let shared_rhs_pos = shared_rhs_tile * smem_stride;

        let lhs_slice = shared_memories
            .lhs
            .slice(shared_lhs_pos, shared_lhs_pos + smem_stride);
        let rhs_slice = shared_memories
            .rhs
            .slice(shared_rhs_pos, shared_rhs_pos + smem_stride);

        cmma::load::<FC>(&lhs, lhs_slice, UInt::new(16));
        cmma::load::<FC>(&rhs, rhs_slice, UInt::new(16));

        cmma::execute::<FC, FC, F, F>(&lhs, &rhs, &accumulator, &accumulator);
    }
}
