use cubecl_core as cubecl;
use cubecl_core::{CubeType, prelude::*};

use crate::matmul::components::{stage::StageConfig, tile};

use super::AccumulatorLoader;

/// Accumulator loader that zeros the accumulator
#[derive(CubeType)]
pub struct ZeroAccumulatorLoader;

#[cube]
impl<O: Numeric, Acc: Numeric, G: StageConfig> AccumulatorLoader<O, Acc, G>
    for ZeroAccumulatorLoader
{
    fn fill_stage(_this: &mut Self, #[comptime] _config: G) {}

    fn load<I: Numeric, Tile: tile::TileMatmul<I, Acc>>(
        _this: &mut Self,
        acc: &mut Tile::Accumulator,
        _n_tile: u32,
        #[comptime] config: Tile::Config,
    ) {
        Tile::zero_accumulator(acc, config);
    }
}
