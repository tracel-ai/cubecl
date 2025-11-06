use crate::components::MatmulPrecision;
use crate::components::global;
use crate::components::stage::StageConfig;
use crate::components::stage::matmul::partition::{Accumulators, PartitionMatmul, RhsTile};
use crate::components::stage::matmul::scheduler::PartitionScheduler;
use crate::components::stage::{NoEvent, StageEventListener};
use crate::components::tile::TileMatmul;
use crate::components::{MatrixPrecision, stage::Stage};
use crate::components::{global::WriteEventListener, stage::StageMatmul};
use core::marker::PhantomData;
use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::tensor::layout::Coords2d;

#[cube]
/// Defines how the stage is partitioned among compute primitives (e.g., units or planes).
/// Controls global writeback and and compute indexing.
pub trait StagePartitioner: Send + Sync + 'static {
    /// Returns the (row, col) of the current compute primitive within the stage.
    fn coordinates<S: StageConfig>(#[comptime] config: S) -> Coords2d;
}

/// Stage Matmul implementation that splits its stage across partitions, one per compute primitive.
///
/// Its results are written in a temporary shared memory to correct the layout before storing to global memory.
pub struct PartitionedStageMatmul<
    MP: MatmulPrecision,
    TM: TileMatmul<
            <MP::Lhs as MatrixPrecision>::Register,
            <MP::Rhs as MatrixPrecision>::Register,
            <MP::Acc as MatrixPrecision>::Register,
        >,
    StageLhs: Stage<
            <<MP as MatmulPrecision>::Lhs as MatrixPrecision>::Stage,
            ReadOnly,
            TileKind = TM::LhsTile,
        >,
    StageRhs: Stage<
            <<MP as MatmulPrecision>::Rhs as MatrixPrecision>::Stage,
            ReadOnly,
            TileKind = TM::RhsTile,
        >,
    StageAcc: Stage<
            <<MP as MatmulPrecision>::Acc as MatrixPrecision>::Stage,
            ReadOnly,
            TileKind = TM::AccTile,
        >,
    StageOut: Stage<
            <<MP as MatmulPrecision>::Acc as MatrixPrecision>::Stage,
            ReadWrite,
            TileKind = TM::OutTile,
        >,
    SP: StagePartitioner,
    S: StageConfig<TileConfig = TM::Config>,
> {
    #[allow(clippy::type_complexity)]
    _phantom: PhantomData<(MP, TM, StageLhs, StageRhs, StageAcc, StageOut, SP, S)>,
}

#[cube]
impl<MP, TM, StageLhs, StageRhs, StageAcc, StageOut, SP, S> StageMatmul<MP>
    for PartitionedStageMatmul<MP, TM, StageLhs, StageRhs, StageAcc, StageOut, SP, S>
where
    MP: MatmulPrecision,
    TM: TileMatmul<
            <MP::Lhs as MatrixPrecision>::Register,
            <MP::Rhs as MatrixPrecision>::Register,
            <MP::Acc as MatrixPrecision>::Register,
        >,
    StageLhs: Stage<
            <<MP as MatmulPrecision>::Lhs as MatrixPrecision>::Stage,
            ReadOnly,
            TileKind = TM::LhsTile,
        >,
    StageRhs: Stage<
            <<MP as MatmulPrecision>::Rhs as MatrixPrecision>::Stage,
            ReadOnly,
            TileKind = TM::RhsTile,
        >,
    StageAcc: Stage<
            <<MP as MatmulPrecision>::Acc as MatrixPrecision>::Stage,
            ReadOnly,
            TileKind = TM::AccTile,
        >,
    StageOut: Stage<
            <<MP as MatmulPrecision>::Acc as MatrixPrecision>::Stage,
            ReadWrite,
            TileKind = TM::OutTile,
        >,
    SP: StagePartitioner,
    S: StageConfig<TileConfig = TM::Config>,
{
    type Config = S;

    type LhsStage = StageLhs;
    type RhsStage = StageRhs;
    type AccStage = StageAcc;
    type OutStage = StageOut;

    type Accumulators = Accumulators<MP, TM, S>;
    type LhsTile = Sequence<TM::LhsFragment>;
    type RhsTile = RhsTile<TM::RhsFragment>;

    fn execute(
        lhs_stage: &StageLhs,
        rhs_stage: &StageRhs,
        lhs_fragment: &mut Self::LhsTile,
        rhs_fragments: &mut Self::RhsTile,
        acc: &mut Self::Accumulators,
        #[comptime] config: Self::Config,
        partition_scheduler: &PartitionScheduler,
    ) {
        Self::execute_with_listener::<NoEvent>(
            lhs_stage,
            rhs_stage,
            lhs_fragment,
            rhs_fragments,
            acc,
            config,
            NoEvent::new(),
            partition_scheduler,
        )
    }

    fn execute_with_listener<SEL: StageEventListener<Self::Config>>(
        lhs_stage: &StageLhs,
        rhs_stage: &StageRhs,
        lhs_fragment: &mut Self::LhsTile,
        rhs_fragments: &mut Self::RhsTile,
        acc: &mut Self::Accumulators,
        #[comptime] config: Self::Config,
        listener: SEL,
        partition_scheduler: &PartitionScheduler,
    ) {
        PartitionMatmul::<MP, TM, StageLhs, StageRhs, StageAcc, S>::execute_with_listener::<SEL>(
            lhs_stage,
            rhs_stage,
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
        stage: &Self::AccStage,
        acc: &mut Self::Accumulators,
        #[comptime] config: Self::Config,
    ) {
        PartitionMatmul::<MP, TM, StageLhs, StageRhs, StageAcc, S>::load_accumulator(
            stage, acc, config,
        );
    }

    fn write_results<W: WriteEventListener, G: global::GlobalConfig>(
        acc: &Self::Accumulators,
        stage: &mut Self::OutStage,
        listener: &mut W,
        partition_scheduler: &PartitionScheduler,
        #[comptime] stage_config: Self::Config,
        #[comptime] global_config: G,
    ) {
        let m_iterations = global_config.tiling_scheme().tiles_in_stage_partition_m();
        let n_iterations = global_config.tiling_scheme().tiles_in_stage_partition_n();

        W::on_event(listener, global::WriteEvent::new_Begin());

        // Iterate over each tile in the partition
        #[unroll]
        for m_iter in 0..m_iterations {
            let m_load_iter = partition_scheduler.map_m(m_iter);

            #[unroll]
            for n_iter in 0..n_iterations {
                let n_load_iter = partition_scheduler.map_n(n_iter);

                let tile_accumulator =
                    Accumulators::<MP, TM, S>::get_at(acc, m_iter, n_iter, stage_config);

                let tile_pos = (m_load_iter, n_load_iter);
                let mut tile = Self::OutStage::tile(stage, tile_pos);

                // Write the results for one tile. To save shared memory space, it reuses the same spot for
                // all tiles in the partition
                TM::write_results(&mut tile, tile_accumulator, stage_config.tile_config());
                W::on_event(listener, global::WriteEvent::new_TileStored(tile_pos));
            }
        }

        W::on_event(listener, global::WriteEvent::new_Finish());
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
