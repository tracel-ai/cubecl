use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use std::marker::PhantomData;

use crate::matmul::interface::performer::Unit;

use super::base::TileMatmul;
/// Does not use specialized hardware
pub struct OuterProductSumTileMatmul<N: Numeric> {
    _n: PhantomData<N>,
}

#[derive(CubeType, Copy, Clone)]
pub struct OuterProductSumTileConfig {
    tile_m: u32,
    tile_k: u32,
    tile_n: u32,
    unroll: bool,
}

#[derive(CubeType)]
pub struct OPSMemory<N: Numeric> {
    smem: SharedMemory<N>,
    smem_stride: u32,
}

#[cube]
pub fn read<N: Numeric>(ops_memory: &OPSMemory<N>, dot_index: u32) -> N {
    ops_memory.smem[(unit_row + dot_index * ops_memory.smem_stride) / ops_memory.smem.line_size()];
}

#[cube]
impl<N: Numeric> TileMatmul for OuterProductSumTileMatmul<N> {
    type Performer = Unit;
    type Input = OPSMemory<N>;
    type Accumulator = Array<N>;
    type Config = OuterProductSumTileConfig;

    fn execute(
        lhs: &Self::Input,
        rhs: &Self::Input,
        acc: &mut Self::Accumulator,
        #[comptime] c: Self::Config,
    ) {
        #[unroll(c.unroll)]
        for dot_index in 0..c.tile_k {
            let register_m = read(lhs, dot_index);
            let register_n = read(rhs, dot_index);

            tile_outer_product(register_m, register_n, acc, c);
        }
    }
}

#[cube]
pub(crate) fn tile_outer_product<N: Numeric>(
    register_m: N,          // line size must equal config.tile_m
    register_n: N,          // line size must equal config.tile_n
    results: &mut Array<N>, // length must equal config.tile_m * config.tile_n
    #[comptime] c: OuterProductSumTileConfig,
) {
    #[unroll(c.unroll)]
    for idx_m in 0..c.tile_m {
        let res_pos_base = idx_m * c.tile_n;
        let value_m = register_m[idx_m];

        #[unroll(c.unroll)]
        for idx_n in 0..c.tile_n {
            let mul = value_m * register_n[idx_n];
            results[res_pos_base + idx_n] += mul;
        }
    }
}
