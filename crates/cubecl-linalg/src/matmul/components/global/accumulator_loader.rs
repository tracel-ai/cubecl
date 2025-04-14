use cubecl_core as cubecl;
use cubecl_core::{CubeType, prelude::*};

use crate::matmul::components::MatmulPrecision;
use crate::matmul::components::{stage::StageConfig, tile};

use super::AccumulatorLoader;

/// Accumulator loader that zeros the accumulator
#[derive(CubeType)]
pub struct ZeroAccumulatorLoader;

#[cube]
impl<MP: MatmulPrecision> AccumulatorLoader<MP> for ZeroAccumulatorLoader {
    fn fill_stage<S: StageConfig>(_this: &mut Self, #[comptime] _config: S) {}

    fn load<Tile: tile::TileMatmul<MP>>(
        _this: &mut Self,
        acc: &mut Tile::Accumulator,
        _n_tile: u32,
        #[comptime] config: Tile::Config,
    ) {
        Tile::zero_accumulator(acc, config);
    }
}
