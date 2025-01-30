use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::config::CubeTiling2dConfig;

#[cube]
pub(crate) fn tile_outer_product<N: Numeric>(
    register_m: Line<N>,
    register_n: Line<N>,
    results: &mut Array<N>,
    #[comptime] config: CubeTiling2dConfig,
) {
    let tile_size = config.tile_size;
    let unroll = config.unroll_tile;

    #[unroll(unroll)]
    for res_idx_m in 0..register_m.size() {
        let res_pos_base = res_idx_m * tile_size;
        #[unroll(unroll)]
        for res_idx_n in 0..register_n.size() {
            let mul: N = register_m[res_idx_m] * register_n[res_idx_n];
            results[res_pos_base + res_idx_n] += mul;
        }
    }
}
