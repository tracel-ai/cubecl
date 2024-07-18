use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{base::Coordinates, config::CubeTiling2dConfig, outer_product::tile_outer_product};

#[cube]
#[allow(unused_mut)]
pub(crate) fn compute_loop<F: Float>(
    coordinates: Coordinates,
    shared_lhs: SharedMemory<F>,
    shared_rhs: SharedMemory<F>,
    results: &mut Array<F>,
    config: Comptime<CubeTiling2dConfig>,
) {
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let block_size_m = Comptime::map(config, |c| c.block_size_m);
    let block_size_k = Comptime::runtime(Comptime::map(config, |c| c.block_size_k));
    let block_size_n = Comptime::map(config, |c| c.block_size_n);
    let unroll = Comptime::map(config, |c| c.unroll_compute);

    let unit_row = coordinates.unit_row;
    let unit_col = coordinates.unit_col;

    for dot_index in range(0u32, block_size_k, unroll) {
        let register_m = shared_lhs[(unit_row + dot_index * Comptime::runtime(block_size_m))
            / Comptime::runtime(tile_size)];
        let register_n = shared_rhs[(unit_col + dot_index * Comptime::runtime(block_size_n))
            / Comptime::runtime(tile_size)];

        tile_outer_product::<F>(register_m, register_n, results, config);
    }
}