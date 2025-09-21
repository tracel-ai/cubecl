use std::marker::PhantomData;

use crate::components::{AccS, stage::StageReader, tile::TileMatmul};
use crate::components::{InputPrecision, MatmulPrecision};
use crate::components::{stage::StageConfig, tile::reader::ReaderKind};
use cubecl::prelude::*;
use cubecl_core::{self as cubecl, intrinsic};

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
    sequence: Sequence<TM::AccFragment>,
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

    /// Load all accumulators from the specified reader
    pub fn load<R: StageReader<AccS<MP>, TileKind = ReaderKind<TM::AccTileReader>>>(
        &mut self,
        reader: &R,
        #[comptime] config: S,
    ) {
        let size_m = comptime![config.tiling_scheme().tiles_in_stage_partition_m()];
        let size_n = comptime![config.tiling_scheme().tiles_in_stage_partition_n()];
        #[unroll]
        for m in 0..size_m {
            #[unroll]
            for n in 0..size_n {
                let acc = self.get_at_mut(unwrap(m), unwrap(n), config);
                let tile = R::read_tile::<S::StageMemoryConfig>(
                    reader,
                    m,
                    n,
                    config.stage_memory_config(),
                );
                TM::load_acc(tile, acc, config.tile_config());
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

#[cube]
#[allow(unused)]
fn unwrap(i: u32) -> comptime_type!(u32) {
    intrinsic!(|_| i.constant().unwrap().as_u32())
}
