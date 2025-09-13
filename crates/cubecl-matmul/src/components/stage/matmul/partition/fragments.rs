use std::marker::PhantomData;

use crate::components::{AccS, stage::StageReader, tile::TileMatmul};
use crate::components::{InputPrecision, MatmulPrecision};
use crate::components::{stage::StageConfig, tile::loader::LoaderKind};
use cubecl::prelude::*;
use cubecl_core as cubecl;

#[derive(CubeType)]
/// Wrapper over a sequence of Tile Matmul accumulators
/// Enables indexing at 2d coordinates
pub struct Accumulators<
    MP: MatmulPrecision,
    TM: TileMatmul<
            <MP::Lhs as InputPrecision>::Register,
            <MP::Rhs as InputPrecision>::Register,
            <MP::Acc as InputPrecision>::Register,
        >,
    S: StageConfig<TileConfig = TM::Config>,
> {
    sequence: Sequence<TM::Accumulator>,
    #[cube(comptime)]
    _phantom: PhantomData<S>,
}

#[cube]
impl<
    MP: MatmulPrecision,
    TM: TileMatmul<
            <MP::Lhs as InputPrecision>::Register,
            <MP::Rhs as InputPrecision>::Register,
            <MP::Acc as InputPrecision>::Register,
        >,
    S: StageConfig<TileConfig = TM::Config>,
> Accumulators<MP, TM, S>
{
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

    pub fn fill<R: StageReader<AccS<MP>, TileKind = LoaderKind<TM::AccLoader>>>(
        &mut self,
        reader: &R,
        #[comptime] config: S,
    ) {
        let size_m = comptime![config.tiling_scheme().partition_size.m()];
        let size_n = comptime![config.tiling_scheme().partition_size.n()];
        #[unroll]
        for m in 0..size_m {
            let i = m * size_n;
            #[unroll]
            for n in 0..size_n {
                let acc = self.sequence.index_mut(i + n);
                let tile = R::read_tile::<S::StageMemoryConfig>(
                    reader,
                    m,
                    n,
                    config.stage_memory_config(),
                );
                TM::fill_acc(tile, acc, config.tile_config());
            }
        }
    }

    pub fn get_at(
        this: &Accumulators<MP, TM, S>,
        #[comptime] i: u32,
        #[comptime] j: u32,
        #[comptime] config: S,
    ) -> &TM::Accumulator {
        this.sequence.index(comptime!(
            i * config.tiling_scheme().tiles_in_stage_partition_n() + j
        ))
    }

    pub fn get_at_mut(
        this: &mut Accumulators<MP, TM, S>,
        #[comptime] i: u32,
        #[comptime] j: u32,
        #[comptime] config: S,
    ) -> &mut TM::Accumulator {
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
