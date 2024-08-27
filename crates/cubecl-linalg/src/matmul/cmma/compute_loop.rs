use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::base::{coop_id, SharedMemories};
use super::config::CmmaConfig;

#[cube]
#[allow(unused_mut)]
pub(crate) fn compute_loop<F: Float, FC: Float>(
    shared_memories: SharedMemories<FC>,
    accumulators: &mut Sequence<cmma::Matrix<F>>,
    config: Comptime<CmmaConfig>,
) {
    let block_size_m = Comptime::map(config, |c| c.block_size_m);
    let block_size_n = Comptime::map(config, |c| c.block_size_n);
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let lane_dim = Comptime::map(config, |c| c.lane_dim);
    let n_tiles = (block_size_m * block_size_n) / (tile_size * tile_size);
    let num_tiles_in_n = n_tiles / lane_dim;

    let num_coop_per_row = Comptime::runtime((block_size_n / tile_size) / num_tiles_in_n);

    let coop_id = coop_id(config);
    let tile_row = coop_id / num_coop_per_row;
    let tile_col_base = (coop_id % num_coop_per_row) * Comptime::runtime(num_tiles_in_n);

    for n in range(0u32, Comptime::get(num_tiles_in_n), Comptime::new(true)) {
        compute_tile::<F, FC>(
            n,
            tile_row,
            tile_col_base,
            shared_memories,
            *accumulators.index(n),
            config,
        );
    }
}

#[cube]
fn compute_tile<F: Float, FC: Float>(
    n_iter: UInt,
    tile_row: UInt,
    tile_col_base: UInt,
    shared_memories: SharedMemories<FC>,
    accumulator: cmma::Matrix<F>,
    config: Comptime<CmmaConfig>,
) {
    let block_size_k = Comptime::map(config, |c| c.block_size_k);
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let unroll = Comptime::map(config, |c| c.unroll);

    let num_tile_elems = Comptime::runtime(tile_size * tile_size);
    let num_tiles_in_k = Comptime::runtime(block_size_k / tile_size);

    let tile_col = tile_col_base + n_iter;

    for k_iter in range(0u32, num_tiles_in_k, unroll) {
        let shared_lhs_tile = tile_row * num_tiles_in_k + k_iter;
        let shared_rhs_tile = tile_col * num_tiles_in_k + k_iter;
        let shared_lhs_pos = shared_lhs_tile * num_tile_elems;
        let shared_rhs_pos = shared_rhs_tile * num_tile_elems;

        let lhs_slice = shared_memories
            .lhs
            .slice(shared_lhs_pos, shared_lhs_pos + num_tile_elems);
        let rhs_slice = shared_memories
            .rhs
            .slice(shared_rhs_pos, shared_rhs_pos + num_tile_elems);

        let a = cmma::Matrix::<FC>::new(
            cmma::MatrixIdent::A,
            16,
            16,
            16,
            cmma::MatrixLayout::RowMajor,
        );
        let b = cmma::Matrix::<FC>::new(
            cmma::MatrixIdent::B,
            16,
            16,
            16,
            cmma::MatrixLayout::RowMajor,
        );

        cmma::load::<FC>(&a, lhs_slice, UInt::new(16));
        cmma::load::<FC>(&b, rhs_slice, UInt::new(16));

        cmma::execute::<FC, FC, F, F>(&a, &b, &accumulator, &accumulator);
    }
}
