use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::base::{Accumulators, SharedMemories};
use super::config::CmmaConfig;

#[cube]
#[allow(unused_mut)]
pub(crate) fn compute_loop<F: Float, FC: Float>(
    shared_memories: SharedMemories<FC>,
    mut accumulators: Accumulators<F>,
    #[comptime] config: CmmaConfig,
) {
    // Other values not supported
    let n_tiles = 2;

    let block_size_n = config.block_size_n;
    let tile_size = config.tile_size;
    let num_coop_per_row = block_size_n / tile_size / n_tiles;

    let coop_id = UNIT_POS_Y;
    let tile_row = coop_id / num_coop_per_row;
    let tile_col_base = (coop_id % num_coop_per_row) * n_tiles;

    compute_tile::<F, FC>(
        0,
        tile_row,
        tile_col_base,
        shared_memories,
        accumulators.first,
        config,
    );
    compute_tile::<F, FC>(
        1,
        tile_row,
        tile_col_base,
        shared_memories,
        accumulators.second,
        config,
    );
}

#[cube]
fn compute_tile<F: Float, FC: Float>(
    n_iter: u32,
    tile_row: u32,
    tile_col_base: u32,
    shared_memories: SharedMemories<FC>,
    accumulator: cmma::Matrix<F>,
    #[comptime] config: CmmaConfig,
) {
    let block_size_k = config.block_size_k;
    let tile_size = config.tile_size;
    let unroll = config.unroll;

    let num_tile_elems = tile_size * tile_size;
    let k_tiles = block_size_k / tile_size;

    let tile_col = tile_col_base + n_iter;

    #[unroll(unroll)]
    for k_iter in 0..k_tiles {
        let shared_lhs_tile = tile_row * k_tiles + k_iter;
        let shared_rhs_tile = tile_col * k_tiles + k_iter;
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

        cmma::load(&a, lhs_slice, 16);
        cmma::load(&b, rhs_slice, 16);

        cmma::execute::<FC, FC, F, F>(&a, &b, &accumulator, &accumulator);
    }
}
