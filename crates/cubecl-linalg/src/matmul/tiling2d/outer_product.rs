use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::config::CubeTiling2dConfig;

#[cube2]
pub(crate) fn tile_outer_product<F: Float>(
    register_m: F,
    register_n: F,
    results: &mut Array<F>,
    config: Comptime<CubeTiling2dConfig>,
) {
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let unroll = Comptime::map(config, |c| c.unroll_tile);

    for res_idx_m in range(0u32, Comptime::get(tile_size), unroll) {
        let res_pos_base = res_idx_m * Comptime::runtime(tile_size);
        for res_idx_n in range(0u32, Comptime::get(tile_size), unroll) {
            let mul = register_m[res_idx_m] * register_n[res_idx_n];
            results[res_pos_base + res_idx_n] += mul;
        }
    }
}
