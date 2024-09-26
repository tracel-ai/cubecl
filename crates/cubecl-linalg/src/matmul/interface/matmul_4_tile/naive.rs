use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use std::marker::PhantomData;

use crate::matmul::interface::performer::Unit;

use super::base::TileMatmul;
/// Does not use specialized hardware
pub struct NaiveTileMatmul<N: Numeric> {
    _n: PhantomData<N>,
}

#[derive(CubeType, Copy, Clone)]
pub struct NaiveTileConfig {
    tile_m: u32,
    tile_k: u32,
    tile_n: u32,
}

#[cube]
impl<N: Numeric> TileMatmul for NaiveTileMatmul<N> {
    type Performer = Unit;
    type Input = Array<N>;
    type Accumulator = Array<N>;
    type Config = NaiveTileConfig;

    fn execute(
        lhs: &Self::Input,
        rhs: &Self::Input,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        for _ in 0..config.tile_k {
            for i in 0..config.tile_m {
                for j in 0..config.tile_n {
                    acc[i * config.tile_m + j] += lhs[i] * rhs[j]
                }
            }
        }
    }
}
