use crate::components::stage::StageMatmul;
use crate::components::stage::matmul::partition::{Accumulators, PartitionMatmul, RhsTile};
use crate::components::stage::matmul::scheduler::PartitionScheduler;
use crate::components::stage::{NoEvent, StageEventListener};
use crate::components::stage::{RowMajorTilingOrder, StageConfig};
use crate::components::tile::TileMatmul;
use crate::components::{AccG, global::memory::GlobalMemoryConfig};
use crate::components::{AccS, global};
use crate::components::{InputPrecision, stage::Stage};
use crate::components::{MatmulIdent, tile::io::Strided};
use crate::components::{MatmulPrecision, stage::StageMemoryConfig};
use crate::components::{StageIdent, stage::StridedStage};
use crate::components::{global::GlobalWriter, stage::ContiguousTilingLayout};
use core::marker::PhantomData;
use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::tensor::{View, layout::Coords2d};

type StageLayout = ContiguousTilingLayout<RowMajorTilingOrder>;

#[cube]
/// Defines how the stage is partitioned among compute primitives (e.g., units or planes).
/// Controls global writeback and and compute indexing.
pub trait StagePartitioner: Send + Sync + 'static {
    /// Writer used to store accumulators back to global memory.
    type Writer<EO: Numeric>: GlobalWriter<EO, TileKind = Strided>;

    /// Initializes a writer at the given global offsets.
    fn init_writer<EO: Numeric>(
        tensor: View<Line<EO>, Coords2d, ReadWrite>,
        #[comptime] config: GlobalMemoryConfig,
    ) -> Self::Writer<EO>;

    /// Returns the (row, col) of the current compute primitive within the stage.
    fn coordinates<S: StageConfig>(#[comptime] config: S) -> Coords2d;

    /// Returns the stage memory config for the intermediary shared memory.
    fn stage_memory_config<S: StageConfig>(
        #[comptime] config: S,
    ) -> comptime_type!(StageMemoryConfig);
}

/// Stage Matmul implementation that splits its stage across partitions, one per compute primitive.
///
/// Its results are written in a temporary shared memory to correct the layout before storing to global memory.
pub struct PartitionedStageMatmul<
    MP: MatmulPrecision,
    TM: TileMatmul<
            <MP::Lhs as InputPrecision>::Register,
            <MP::Rhs as InputPrecision>::Register,
            <MP::Acc as InputPrecision>::Register,
        >,
    StageLhs: Stage<<<MP as MatmulPrecision>::Lhs as InputPrecision>::Stage, TileKind = TM::LhsTile>,
    StageRhs: Stage<<<MP as MatmulPrecision>::Rhs as InputPrecision>::Stage, TileKind = TM::RhsTile>,
    StageAcc: Stage<<<MP as MatmulPrecision>::Acc as InputPrecision>::Stage, TileKind = TM::AccTile>,
    SP: StagePartitioner,
    S: StageConfig<TileConfig = TM::Config>,
> {
    _phantom: PhantomData<(MP, TM, StageLhs, StageRhs, StageAcc, SP, S)>,
}

#[cube]
impl<MP, TM, StageLhs, StageRhs, StageAcc, SP, S> StageMatmul<MP>
    for PartitionedStageMatmul<MP, TM, StageLhs, StageRhs, StageAcc, SP, S>
where
    MP: MatmulPrecision,
    TM: TileMatmul<
            <MP::Lhs as InputPrecision>::Register,
            <MP::Rhs as InputPrecision>::Register,
            <MP::Acc as InputPrecision>::Register,
            OutTile = Strided,
        >,
    StageLhs:
        Stage<<<MP as MatmulPrecision>::Lhs as InputPrecision>::Stage, TileKind = TM::LhsTile>,
    StageRhs:
        Stage<<<MP as MatmulPrecision>::Rhs as InputPrecision>::Stage, TileKind = TM::RhsTile>,
    StageAcc:
        Stage<<<MP as MatmulPrecision>::Acc as InputPrecision>::Stage, TileKind = TM::AccTile>,
    SP: StagePartitioner,
    S: StageConfig<TileConfig = TM::Config>,
{
    type Config = S;

    type LhsStage = StageLhs;
    type RhsStage = StageRhs;
    type AccStage = StageAcc;
    type Accumulators = Accumulators<MP, TM, S>;
    type LhsTile = Sequence<TM::LhsFragment>;
    type RhsTile = RhsTile<TM::RhsFragment>;
    type GlobalWriter = SP::Writer<AccG<MP>>;

    fn execute(
        lhs_reader: &StageLhs,
        rhs_reader: &StageRhs,
        lhs_fragment: &mut Self::LhsTile,
        rhs_fragments: &mut Self::RhsTile,
        acc: &mut Self::Accumulators,
        #[comptime] config: Self::Config,
        partition_scheduler: &PartitionScheduler,
    ) {
        Self::execute_with_listener::<NoEvent>(
            lhs_reader,
            rhs_reader,
            lhs_fragment,
            rhs_fragments,
            acc,
            config,
            NoEvent::new(),
            partition_scheduler,
        )
    }

    fn execute_with_listener<SEL: StageEventListener<Self::Config>>(
        lhs_reader: &StageLhs,
        rhs_reader: &StageRhs,
        lhs_fragment: &mut Self::LhsTile,
        rhs_fragments: &mut Self::RhsTile,
        acc: &mut Self::Accumulators,
        #[comptime] config: Self::Config,
        listener: SEL,
        partition_scheduler: &PartitionScheduler,
    ) {
        PartitionMatmul::<MP, TM, StageLhs, StageRhs, StageAcc, S>::execute_with_listener::<SEL>(
            lhs_reader,
            rhs_reader,
            lhs_fragment,
            rhs_fragments,
            acc,
            config,
            listener,
            partition_scheduler,
        );
    }

    fn init_tile_inputs(#[comptime] config: Self::Config) -> (Self::LhsTile, Self::RhsTile) {
        PartitionMatmul::<MP, TM, StageLhs, StageRhs, StageAcc, S>::init_tile_inputs(config)
    }

    fn init_accumulators(#[comptime] config: Self::Config) -> Self::Accumulators {
        PartitionMatmul::<MP, TM, StageLhs, StageRhs, StageAcc, S>::init_accumulator(config)
    }

    fn load_accumulators(
        reader: &Self::AccStage,
        acc: &mut Self::Accumulators,
        #[comptime] config: Self::Config,
    ) {
        PartitionMatmul::<MP, TM, StageLhs, StageRhs, StageAcc, S>::load_accumulator(
            reader, acc, config,
        );
    }

    fn write_results<G: global::GlobalConfig>(
        acc: &Accumulators<MP, TM, S>,
        out: &mut Self::GlobalWriter,
        partition_scheduler: &PartitionScheduler,
        #[comptime] stage_config: S,
        #[comptime] global_config: G,
    ) {
        let out_stage = StridedStage::<AccS<MP>, StageLayout>::new(
            StageIdent::Acc,
            SP::stage_memory_config::<S>(stage_config),
        );

        let partition = SP::coordinates::<Self::Config>(stage_config);
        let mut out_tile = out_stage.get_tile_mut(partition);

        let m_iterations = global_config.tiling_scheme().tiles_in_stage_partition_m();
        let n_iterations = global_config.tiling_scheme().tiles_in_stage_partition_n();

        let mut m_iter = comptime![0u32];

        // Iterate over each tile in the partition
        #[unroll]
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..comptime![m_iterations] {
            let m_load_iter = partition_scheduler.map_m(m_iter);
            let mut n_iter = comptime![0u32];

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..comptime![n_iterations] {
                let n_load_iter = partition_scheduler.map_n(n_iter);

                let tile_accumulator =
                    Accumulators::<MP, TM, S>::get_at(acc, m_iter, n_iter, stage_config);

                // Write the results for one tile. To save shared memory space, it reuses the same spot for
                // all tile in the partition
                TM::write_results(&mut out_tile, tile_accumulator, stage_config.tile_config());

                // Write the current tile result to global memory
                Self::GlobalWriter::write(
                    out,
                    &out_tile,
                    (m_load_iter, n_load_iter),
                    global_config.plane_dim(),
                    global_config.global_memory_config(MatmulIdent::Out),
                );

                comptime![n_iter += 1];
            }
            comptime![m_iter += 1];
        }
    }

    fn init_writer(
        tensor: View<Line<AccG<MP>>, Coords2d, ReadWrite>,
        #[comptime] config: GlobalMemoryConfig,
    ) -> Self::GlobalWriter {
        SP::init_writer(tensor, config)
    }

    fn init_scheduler(#[comptime] config: Self::Config) -> PartitionScheduler {
        let (partition_row, partition_col) = SP::coordinates::<Self::Config>(config);

        PartitionScheduler::new(
            partition_row,
            partition_col,
            config.tiling_scheme().partition_size,
            config.partition_schedule_scheme(),
        )
    }
}
