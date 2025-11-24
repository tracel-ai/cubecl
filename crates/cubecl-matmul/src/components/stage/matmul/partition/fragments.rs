use crate::components::{AccS, stage::Stage, tile::TileMatmul};
use crate::components::{MatmulPrecision, MatrixLayout, MatrixPrecision, PartitionSize};
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
> {
    sequence: Sequence<TM::AccFragment>,
}

#[cube]
impl<
    MP: MatmulPrecision,
    TM: TileMatmul<
            <MP::Lhs as MatrixPrecision>::Register,
            <MP::Rhs as MatrixPrecision>::Register,
            <MP::Acc as MatrixPrecision>::Register,
        >,
> Accumulators<MP, TM>
{
    /// Create a new accumulators sequence from the provided configuration
    pub fn new(
        #[comptime] partition_size: PartitionSize,
        #[comptime] acc_layout: MatrixLayout,
        #[comptime] tile_config: TM::Config,
    ) -> Accumulators<MP, TM> {
        let mut accumulators = Sequence::new();

        #[unroll]
        for _ in 0..comptime!(partition_size.mn()) {
            accumulators.push(TM::allocate_acc(acc_layout, tile_config));
        }

        Accumulators::<MP, TM> {
            sequence: accumulators,
        }
    }

    /// Load all accumulators from the specified stage
    pub fn load<R: Stage<AccS<MP>, ReadOnly, TileKind = TM::AccTile>>(
        &mut self,
        stage: &R,
        #[comptime] tiles_in_stage_partition_m: u32,
        #[comptime] tiles_in_stage_partition_n: u32,
        #[comptime] tile_config: TM::Config,
    ) {
        #[unroll]
        for m in 0..tiles_in_stage_partition_m {
            #[unroll]
            for n in 0..tiles_in_stage_partition_n {
                let acc = self.get_at_mut(m, n, tiles_in_stage_partition_n);
                let tile = R::tile(stage, (m, n).runtime());
                TM::load_acc(&tile, acc, tile_config);
            }
        }
    }

    /// Fetch a reference to the accumulator at (`m`, `n`)
    pub fn get_at(
        &self,
        #[comptime] m: u32,
        #[comptime] n: u32,
        #[comptime] tiles_in_stage_partition_n: u32,
    ) -> &TM::AccFragment {
        self.sequence
            .index(comptime!(m * tiles_in_stage_partition_n + n))
    }

    /// Fetch a mutable reference to the accumulator at (`m`, `n`)
    pub fn get_at_mut(
        &mut self,
        #[comptime] m: u32,
        #[comptime] n: u32,
        #[comptime] tiles_in_stage_partition_n: u32,
    ) -> &mut TM::AccFragment {
        self.sequence
            .index_mut(comptime!(m * tiles_in_stage_partition_n + n))
    }
}

#[derive(CubeType)]
/// Rhs tiles, can be doubled for partition double buffering
pub enum RhsTile<Rhs: CubeType> {
    Single(Rhs),
    Double((Rhs, Rhs)),
}
