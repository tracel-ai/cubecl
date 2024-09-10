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
    #[comptime] comptime_info: ComptimeCmmaInfo,
) {
    let block_size_n = comptime_info.block_size_n;
    let tile_size = comptime_info.tile_size;
    let num_accumulators = comptime_info.num_accumulators;
    let num_coop_per_row = (block_size_n / tile_size) / num_accumulators;

    let tile_row = ids.coop / num_coop_per_row;
    let tile_col_base = (ids.coop % num_coop_per_row) * num_accumulators;

    let lhs = &cmma_matrices.lhs;
    let rhs = &cmma_matrices.rhs;
    let accumulators = &cmma_matrices.accumulators;

    #[unroll]
    for n in 0..num_accumulators {
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
    tile_row: u32,
    tile_col: u32,
    shared_memories: SharedMemories<FC>,
    lhs: &cmma::Matrix<FC>,
    rhs: &cmma::Matrix<FC>,
    accumulator: &cmma::Matrix<F>,
    #[comptime] comptime_info: ComptimeCmmaInfo,
) {
    let block_size_k = comptime_info.block_size_k;
    let tile_size = comptime_info.tile_size;
    let unroll = comptime_info.unroll;

    let smem_stride = tile_size * tile_size;
    let num_tiles_in_k = block_size_k / tile_size;

    #[unroll(unroll)]
    for k_iter in 0..num_tiles_in_k {
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

        cmma::load::<FC>(&lhs, lhs_slice, 16);
        cmma::load::<FC>(&rhs, rhs_slice, 16);

        cmma::execute::<FC, FC, F, F>(&lhs, &rhs, &accumulator, &accumulator);
    }
}
