use crate::components::AccG;
use crate::components::MatmulIdent;
use crate::components::MatmulPrecision;
use crate::components::StageIdent;
use crate::components::global;
use crate::components::global::StageUnloader;
use crate::components::stage::StageConfig;
use crate::components::stage::StageMatmul;
use crate::components::stage::matmul::partition::{Accumulators, PartitionMatmul, RhsTile};
use crate::components::stage::matmul::scheduler::PartitionScheduler;
use crate::components::stage::{NoEvent, StageEventListener};
use crate::components::tile::TileMatmul;
use crate::components::tile::loader::LoaderKind;
use crate::components::{InputPrecision, stage::StageReader};
use core::marker::PhantomData;
use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::tensor::{View, layout::Coordinates};

#[cube]
/// Defines how the stage is partitioned among compute primitives (e.g., units or planes).
/// Controls global writeback and and compute indexing.
pub trait StagePartitioner: Send + Sync + 'static {
    /// Writer used to store accumulators back to global memory.
    type Writer<EO: Numeric>: StageUnloader<EO, Coordinates = Self::WriteCoords>;
    /// Coordinates used by the writer
    type WriteCoords: Coordinates;

    /// Initializes a writer at the given global offsets.
    fn init_writer<EO: Numeric>(
        tensor: View<Line<EO>, Self::WriteCoords, ReadWrite>,
    ) -> Self::Writer<EO>;

    /// Returns the (row, col) of the current compute primitive within the stage.
    fn coordinates<S: StageConfig>(#[comptime] config: S) -> (u32, u32);

    /// Returns the total number of compute primitives in the stage.
    fn num_primitives<S: StageConfig>(#[comptime] config: S) -> comptime_type!(u32);
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
    RL: StageReader<
            <<MP as MatmulPrecision>::Lhs as InputPrecision>::Stage,
            TileKind = LoaderKind<TM::LhsTileLoader>,
        >,
    RR: StageReader<
            <<MP as MatmulPrecision>::Rhs as InputPrecision>::Stage,
            TileKind = LoaderKind<TM::RhsTileLoader>,
        >,
    RA: StageReader<
            <<MP as MatmulPrecision>::Acc as InputPrecision>::Stage,
            TileKind = LoaderKind<TM::AccTileLoader>,
        >,
    SP: StagePartitioner,
    S: StageConfig<TileConfig = TM::Config>,
> {
    _phantom: PhantomData<(MP, TM, RL, RR, RA, SP, S)>,
}

#[cube]
impl<MP, TM, RL, RR, RA, SP, S> StageMatmul<MP>
    for PartitionedStageMatmul<MP, TM, RL, RR, RA, SP, S>
where
    MP: MatmulPrecision,
    TM: TileMatmul<
            <MP::Lhs as InputPrecision>::Register,
            <MP::Rhs as InputPrecision>::Register,
            <MP::Acc as InputPrecision>::Register,
        >,
    RL: StageReader<
            <<MP as MatmulPrecision>::Lhs as InputPrecision>::Stage,
            TileKind = LoaderKind<TM::LhsTileLoader>,
        >,
    RR: StageReader<
            <<MP as MatmulPrecision>::Rhs as InputPrecision>::Stage,
            TileKind = LoaderKind<TM::RhsTileLoader>,
        >,
    RA: StageReader<
            <<MP as MatmulPrecision>::Acc as InputPrecision>::Stage,
            TileKind = LoaderKind<TM::AccTileLoader>,
        >,
    SP: StagePartitioner,
    S: StageConfig<TileConfig = TM::Config>,
{
    type Config = S;

    type LhsStageReader = RL;
    type RhsStageReader = RR;
    type AccStageReader = RA;
    type Accumulators = Accumulators<MP, TM, S>;
    type LhsTile = Sequence<TM::LhsFragment>;
    type RhsTile = RhsTile<TM::RhsFragment>;
    type StageUnloader = SP::Writer<AccG<MP>>;
    type WriteCoords = SP::WriteCoords;

    fn execute(
        lhs_reader: &RL,
        rhs_reader: &RR,
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
        lhs_reader: &RL,
        rhs_reader: &RR,
        lhs_fragment: &mut Self::LhsTile,
        rhs_fragments: &mut Self::RhsTile,
        acc: &mut Self::Accumulators,
        #[comptime] config: Self::Config,
        listener: SEL,
        partition_scheduler: &PartitionScheduler,
    ) {
        PartitionMatmul::<MP, TM, RL, RR, RA, S>::execute_with_listener::<SEL>(
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
        PartitionMatmul::<MP, TM, RL, RR, RA, S>::init_tile_inputs(config)
    }

    fn init_accumulators(#[comptime] config: Self::Config) -> Self::Accumulators {
        PartitionMatmul::<MP, TM, RL, RR, RA, S>::init_accumulator(config)
    }

    fn load_accumulators(
        reader: &Self::AccStageReader,
        acc: &mut Self::Accumulators,
        #[comptime] config: Self::Config,
    ) {
        PartitionMatmul::<MP, TM, RL, RR, RA, S>::load_accumulator(reader, acc, config);
    }

    fn write_results<G: global::GlobalConfig>(
        acc: &Accumulators<MP, TM, S>,
        out: &mut Self::StageUnloader,
        partition_scheduler: &PartitionScheduler,
        #[comptime] stage_config: S,
        #[comptime] global_config: G,
    ) {
        let out_smem_line_size = stage_config.stage_line_size(StageIdent::Acc);

        // The out shared memory is one tile wide per partition.
        let num_tile_lines =
            stage_config.tiling_scheme().elements_in_tile_mn() / out_smem_line_size;
        let out_smem_num_lines = num_tile_lines * comptime!(SP::num_primitives(stage_config));

        let m_iterations = global_config.tiling_scheme().tiles_in_stage_partition_m();
        let n_iterations = global_config.tiling_scheme().tiles_in_stage_partition_n();
        let (partition_row, partition_col) = SP::coordinates::<Self::Config>(stage_config);
        let num_partitions_n = stage_config.tiling_scheme().stage_partitions_in_stage_n();

        let mut out_smem =
            SharedMemory::<AccG<MP>>::new_lined(out_smem_num_lines, out_smem_line_size);
        let absolute_partition_position = partition_row * num_partitions_n + partition_col;
        let slice_start = num_tile_lines * absolute_partition_position;
        let mut smem_slice = out_smem.slice_mut(slice_start, slice_start + num_tile_lines);

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
                TM::write_results(
                    tile_accumulator,
                    &mut smem_slice,
                    stage_config.tile_config(),
                );

                // Write the current tile result to global memory
                Self::StageUnloader::write(
                    out,
                    smem_slice.to_slice(),
                    m_load_iter,
                    n_load_iter,
                    out_smem_line_size,
                    global_config.plane_dim(),
                    global_config.global_memory_config(MatmulIdent::Out),
                );

                comptime![n_iter += 1];
            }
            comptime![m_iter += 1];
        }
    }

    fn init_writer(
        tensor: View<Line<AccG<MP>>, Self::WriteCoords, ReadWrite>,
    ) -> Self::StageUnloader {
        SP::init_writer(tensor)
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
