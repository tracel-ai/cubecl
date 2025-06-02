use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{base::Coordinates, config::CubeTiling2dConfig, outer_product::tile_outer_product};

#[cube]
#[allow(unused_mut)]
pub(crate) fn compute_loop<N: Numeric>(
    coordinates: Coordinates,
    shared_lhs: SharedMemory<Line<N>>,
    shared_rhs: SharedMemory<Line<N>>,
    results: &mut Array<N>,
    #[comptime] config: CubeTiling2dConfig,
) {
    let tile_size = config.tile_size;
    let block_size_m = config.block_size_m;
    let block_size_k = config.block_size_k;
    let block_size_n = config.block_size_n;
    let unroll = config.unroll_compute;

    let unit_row = coordinates.unit_row;
    let unit_col = coordinates.unit_col;

    #[unroll(unroll)]
    for dot_index in 0..block_size_k {
        let register_m = shared_lhs[(unit_row + dot_index * block_size_m) / tile_size];
        let register_n = shared_rhs[(unit_col + dot_index * block_size_n) / tile_size];

        tile_outer_product::<N>(register_m, register_n, results, config);
    }
}
