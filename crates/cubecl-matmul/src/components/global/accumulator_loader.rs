use cubecl_core as cubecl;
use cubecl_core::{CubeType, prelude::*};

use crate::components::tile::TileMatmul;
use crate::components::{InputPrecision, MatmulPrecision};

use super::GlobalConfig;

#[cube]
/// Loads an accumulator with pre-defined data
pub trait AccumulatorLoader<MP: MatmulPrecision>: CubeType + 'static + Send + Sync {
    fn fill_stage<G: GlobalConfig>(this: &mut Self, #[comptime] config: G);

    /// Load accumulator for `nth_tile`. Should call either `zero_accumulator` or `fill_accumulator`
    /// for the underlying tile.
    fn load<
        TM: TileMatmul<
                <MP::Lhs as InputPrecision>::Register,
                <MP::Rhs as InputPrecision>::Register,
                MP::EA,
            >,
    >(
        this: &mut Self,
        acc: &mut TM::Accumulator,
        nth_tile: u32,
        #[comptime] config: TM::Config,
    );
}

#[derive(CubeType)]
/// Accumulator loader that zeros the accumulator
pub struct ZeroAccumulatorLoader;

#[cube]
impl<MP: MatmulPrecision> AccumulatorLoader<MP> for ZeroAccumulatorLoader {
    fn fill_stage<G: GlobalConfig>(_this: &mut Self, #[comptime] _config: G) {}

    fn load<
        Tile: TileMatmul<
                <MP::Lhs as InputPrecision>::Register,
                <MP::Rhs as InputPrecision>::Register,
                MP::EA,
            >,
    >(
        _this: &mut Self,
        acc: &mut Tile::Accumulator,
        _n_tile: u32,
        #[comptime] config: Tile::Config,
    ) {
        Tile::zero_accumulator(acc, config);
    }
}
