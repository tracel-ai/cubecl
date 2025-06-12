use crate::components::MatmulPrecision;
use crate::components::global::AccumulatorLoader;
use crate::components::stage::StageConfig;
use crate::components::stage::matmul::config::PartitionedStageConfig;
use crate::components::tile::TileMatmul;
use cubecl::prelude::*;
use cubecl_core as cubecl;

#[derive(CubeType)]
/// Wrapper over a sequence of tile matmul accumulators
/// Enables indexing at 2d coordinates
pub struct Accumulators<MP: MatmulPrecision, TMM: TileMatmul<MP>> {
    sequence: Sequence<TMM::Accumulator>,
}

#[cube]
impl<MP: MatmulPrecision, TMM: TileMatmul<MP>> Accumulators<MP, TMM> {
    pub fn new(#[comptime] config: PartitionedStageConfig<TMM::Config>) -> Accumulators<MP, TMM> {
        let partition_size = config.tiling_scheme().partition_size;
        let mut accumulators = Sequence::new();

        #[unroll]
        for _ in 0..comptime!(partition_size.mn()) {
            accumulators.push(TMM::allocate_accumulator(config.tile_config()));
        }

        Accumulators::<MP, TMM> {
            sequence: accumulators,
        }
    }

    pub fn zero(&mut self, #[comptime] config: PartitionedStageConfig<TMM::Config>) {
        #[unroll]
        for i in 0..comptime![config.tiling_scheme().partition_size.mn()] {
            TMM::zero_accumulator(self.sequence.index_mut(i), config.tile_config());
        }
    }

    pub fn fill<L: AccumulatorLoader<MP>>(
        &mut self,
        loader: &mut L,
        #[comptime] config: PartitionedStageConfig<TMM::Config>,
    ) {
        #[unroll]
        for i in 0..comptime![config.tiling_scheme().partition_size.mn()] {
            let acc = self.sequence.index_mut(i);
            L::load::<TMM>(loader, acc, i, config.tile_config());
        }
    }

    pub fn get_at(
        this: &Accumulators<MP, TMM>,
        #[comptime] i: u32,
        #[comptime] j: u32,
        #[comptime] config: PartitionedStageConfig<TMM::Config>,
    ) -> &TMM::Accumulator {
        this.sequence.index(comptime!(
            i * config.tiling_scheme().tiles_in_stage_partition_n() + j
        ))
    }

    pub fn get_at_mut(
        this: &mut Accumulators<MP, TMM>,
        #[comptime] i: u32,
        #[comptime] j: u32,
        #[comptime] config: PartitionedStageConfig<TMM::Config>,
    ) -> &mut TMM::Accumulator {
        this.sequence.index_mut(comptime!(
            i * config.tiling_scheme().tiles_in_stage_partition_n() + j
        ))
    }
}

#[derive(CubeType)]
pub enum RhsTile<Rhs: CubeType> {
    Single(Rhs),
    Double((Rhs, Rhs)),
}
