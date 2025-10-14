use std::marker::PhantomData;

use crate::components::stage::StageConfig;
use crate::components::{AccS, stage::Stage, tile::TileMatmul};
use crate::components::{MatmulPrecision, MatrixPrecision};
use cubecl::prelude::*;
use cubecl_core::{self as cubecl};

#[derive(CubeType)]
/// Wrapper over a sequence of Tile Matmul accumulators
/// Enables indexing at 2d coordinates
pub struct Accumulators<
    MP: MatmulPrecision,
    TM: TileMatmul<
            <MP::Lhs as MatrixPrecision>::Register,
            <MP::Rhs as MatrixPrecision>::Register,
            <MP::Acc as MatrixPrecision>::Register,
        >,
    S: StageConfig<TileConfig = TM::Config>,
> {
    sequence: Sequence<TM::AccFragment>,
    #[cube(comptime)]
    _phantom: PhantomData<S>,
}

#[cube]
impl<
    MP: MatmulPrecision,
    TM: TileMatmul<
            <MP::Lhs as MatrixPrecision>::Register,
            <MP::Rhs as MatrixPrecision>::Register,
            <MP::Acc as MatrixPrecision>::Register,
        >,
    S: StageConfig<TileConfig = TM::Config>,
> Accumulators<MP, TM, S>
{
    /// Create a new accumulators sequence from the provided configuration
    pub fn new(#[comptime] config: S) -> Accumulators<MP, TM, S> {
        let partition_size = config.tiling_scheme().partition_size;
        let mut accumulators = Sequence::new();

        #[unroll]
        for _ in 0..comptime!(partition_size.mn()) {
            accumulators.push(TM::allocate_acc(config.tile_config()));
        }

        Accumulators::<MP, TM, S> {
            sequence: accumulators,
            _phantom: PhantomData,
        }
    }

    /// Load all accumulators from the specified stage
    pub fn load<R: Stage<AccS<MP>, ReadOnly, TileKind = TM::AccTile>>(
        &mut self,
        stage: &R,
        #[comptime] config: S,
    ) {
        let size_m = comptime![config.tiling_scheme().tiles_in_stage_partition_m()];
        let size_n = comptime![config.tiling_scheme().tiles_in_stage_partition_n()];
        #[unroll]
        for m in 0..size_m {
            #[unroll]
            for n in 0..size_n {
                let acc = self.get_at_mut(m, n, config);
                let tile = R::tile(stage, (m, n).runtime());
                TM::load_acc(&tile, acc, config.tile_config());
            }
        }
    }

    /// Fetch a reference to the accumulator at (`m`, `n`)
    pub fn get_at(
        &self,
        #[comptime] m: u32,
        #[comptime] n: u32,
        #[comptime] config: S,
    ) -> &TM::AccFragment {
        self.sequence.index(comptime!(
            m * config.tiling_scheme().tiles_in_stage_partition_n() + n
        ))
    }

    /// Fetch a mutable reference to the accumulator at (`m`, `n`)
    pub fn get_at_mut(
        &mut self,
        #[comptime] m: u32,
        #[comptime] n: u32,
        #[comptime] config: S,
    ) -> &mut TM::AccFragment {
        self.sequence.index_mut(comptime!(
            m * config.tiling_scheme().tiles_in_stage_partition_n() + n
        ))
    }
}

#[derive(CubeType)]
/// Rhs tiles, can be doubled for partition double buffering
pub enum RhsTile<Rhs: CubeType> {
    Single(Rhs),
    Double((Rhs, Rhs)),
}
