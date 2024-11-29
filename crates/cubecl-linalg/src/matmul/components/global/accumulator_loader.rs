use cubecl_core as cubecl;
use cubecl_core::{prelude::*, CubeType};

use crate::matmul::components::tile;

use super::AccumulatorLoader;

/// Accumulator loader that zeros the accumulator
#[derive(CubeType)]
pub struct ZeroAccumulatorLoader;

#[cube]
impl<O: Numeric, Acc: Numeric> AccumulatorLoader<O, Acc> for ZeroAccumulatorLoader {
    fn fill_stage<G>(_this: &mut Self, #[comptime] _config: G) {}

    fn load<I: Numeric, Tile: tile::Matmul<I, Acc>>(
        _this: &mut Self,
        acc: &mut Tile::Accumulator,
        _n_tile: u32,
        #[comptime] config: Tile::Config,
    ) {
        Tile::zero_accumulator(acc, config);
    }
}
