use std::marker::PhantomData;

use crate::components::MatmulPrecision;
use crate::components::global::AccumulatorLoader;
use crate::components::stage::StageConfig;
use crate::components::tile::TileMatmul;
use cubecl::prelude::*;
use cubecl_core as cubecl;

#[derive(CubeType)]
/// Wrapper over a sequence of Tile Matmul accumulators
/// Enables indexing at 2d coordinates
pub struct Accumulators<
    MP: MatmulPrecision,
    TMM: TileMatmul<MP>,
    S: StageConfig<TileConfig = TMM::Config>,
> {
    sequence: Sequence<TMM::Accumulator>,
    #[cube(comptime)]
    _phantom: PhantomData<S>,
}

#[cube]
impl<MP: MatmulPrecision, TMM: TileMatmul<MP>, S: StageConfig<TileConfig = TMM::Config>>
    Accumulators<MP, TMM, S>
{
    pub fn new(#[comptime] config: S) -> Accumulators<MP, TMM, S> {
        let partition_size = config.tiling_scheme().partition_size;
        let mut accumulators = Sequence::new();

        #[unroll]
        for _ in 0..comptime!(partition_size.mn()) {
            accumulators.push(TMM::allocate_accumulator(config.tile_config()));
        }

        Accumulators::<MP, TMM, S> {
            sequence: accumulators,
            _phantom: PhantomData,
        }
    }

    pub fn zero(&mut self, #[comptime] config: S) {
        #[unroll]
        for i in 0..comptime![config.tiling_scheme().partition_size.mn()] {
            TMM::zero_accumulator(self.sequence.index_mut(i), config.tile_config());
        }
    }

    pub fn fill<L: AccumulatorLoader<MP>>(&mut self, loader: &mut L, #[comptime] config: S) {
        #[unroll]
        for i in 0..comptime![config.tiling_scheme().partition_size.mn()] {
            let acc = self.sequence.index_mut(i);
            L::load::<TMM>(loader, acc, i, config.tile_config());
        }
    }

    pub fn get_at(
        this: &Accumulators<MP, TMM, S>,
        #[comptime] i: u32,
        #[comptime] j: u32,
        #[comptime] config: S,
    ) -> &TMM::Accumulator {
        this.sequence.index(comptime!(
            i * config.tiling_scheme().tiles_in_stage_partition_n() + j
        ))
    }

    pub fn get_at_mut(
        this: &mut Accumulators<MP, TMM, S>,
        #[comptime] i: u32,
        #[comptime] j: u32,
        #[comptime] config: S,
    ) -> &mut TMM::Accumulator {
        this.sequence.index_mut(comptime!(
            i * config.tiling_scheme().tiles_in_stage_partition_n() + j
        ))
    }
}

#[derive(CubeType)]
/// Rhs tiles, can be doubled for partition double buffering
pub enum RhsTile<Rhs: CubeType> {
    Single(Rhs),
    Double((Rhs, Rhs)),
}
