use crate::components::InputPrecision;
use crate::components::LhsS;
use crate::components::MatmulPrecision;
use crate::components::PartitionSize;
use crate::components::RhsS;
use crate::components::StageIdent;
use crate::components::global;
use crate::components::global::AccumulatorLoader;
use crate::components::global::GlobalWriter;
use crate::components::stage::StageConfig;
use crate::components::stage::StageMatmul;
use crate::components::stage::StageToTileReader;
use crate::components::stage::matmul::partition::{Accumulators, PartitionMatmul, RhsTile};
use crate::components::stage::{NoEvent, StageEventListener};
use crate::components::tile::TileMatmul;
use core::marker::PhantomData;
use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};

#[cube]
/// Defines how the stage is partitioned among compute primitives (e.g., units or planes).
/// Controls global writeback and and compute indexing.
pub trait StagePartitioner: Send + Sync + 'static {
    /// Writer used to store accumulators back to global memory.
    type Writer<EO: Numeric>: GlobalWriter<EO>;

    /// Initializes a writer at the given global offsets.
    fn init_writer<EO: Numeric>(
        tensor: VirtualTensor<EO, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
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
            MP::EA,
        >,
    RL: StageToTileReader<LhsS<MP>>,
    RR: StageToTileReader<RhsS<MP>>,
    SP: StagePartitioner,
    S: StageConfig<TileConfig = TM::Config>,
> {
    _phantom: PhantomData<(MP, TM, RL, RR, SP, S)>,
}

#[cube]
impl<MP, TM, RL, RR, SP, S> StageMatmul<MP> for PartitionedStageMatmul<MP, TM, RL, RR, SP, S>
where
    MP: MatmulPrecision,
    TM: TileMatmul<
            <MP::Lhs as InputPrecision>::Register,
            <MP::Rhs as InputPrecision>::Register,
            MP::EA,
        >,
    RL: StageToTileReader<<<MP as MatmulPrecision>::Lhs as InputPrecision>::Stage>,
    RR: StageToTileReader<<<MP as MatmulPrecision>::Rhs as InputPrecision>::Stage>,
    SP: StagePartitioner,
    S: StageConfig<TileConfig = TM::Config>,
{
    type Config = S;

    type LhsReader = RL;
    type RhsReader = RR;
    type Accumulator = Accumulators<MP, TM, S>;
    type LhsTile = Sequence<TM::Lhs>;
    type RhsTile = RhsTile<TM::Rhs>;
    type Writer = SP::Writer<MP::EO>;

    fn execute(
        lhs_reader: &RL,
        rhs_reader: &RR,
        lhs_fragment: &mut Self::LhsTile,
        rhs_fragments: &mut Self::RhsTile,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        Self::execute_with_listener::<NoEvent>(
            lhs_reader,
            rhs_reader,
            lhs_fragment,
            rhs_fragments,
            acc,
            config,
            NoEvent::new(),
        )
    }

    fn execute_with_listener<SEL: StageEventListener<Self::Config>>(
        lhs_reader: &RL,
        rhs_reader: &RR,
        lhs_fragment: &mut Self::LhsTile,
        rhs_fragments: &mut Self::RhsTile,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
        listener: SEL,
    ) {
        let (partition_row, partition_col) = SP::coordinates::<Self::Config>(config);
        let partition_scheduler = PartitionScheduler::new(
            partition_row,
            partition_col,
            config.tiling_scheme().partition_size,
            PartitionScheduleScheme::Naive,
        );

        PartitionMatmul::<MP, TM, RL, RR, S>::execute_with_listener::<SEL>(
            partition_scheduler,
            lhs_reader,
            rhs_reader,
            lhs_fragment,
            rhs_fragments,
            acc,
            config,
            listener,
        );
    }

    fn init_tile_inputs(#[comptime] config: Self::Config) -> (Self::LhsTile, Self::RhsTile) {
        PartitionMatmul::<MP, TM, RL, RR, S>::init_tile_inputs(config)
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        PartitionMatmul::<MP, TM, RL, RR, S>::init_accumulator(config)
    }

    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] config: Self::Config) {
        PartitionMatmul::<MP, TM, RL, RR, S>::zero_accumulator(acc, config);
    }

    fn fill_accumulator<L: AccumulatorLoader<MP>>(
        loader: &mut L,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        PartitionMatmul::<MP, TM, RL, RR, S>::fill_accumulator::<L>(loader, acc, config);
    }

    fn write_results<G: global::GlobalConfig>(
        acc: &Accumulators<MP, TM, S>,
        out: &mut Self::Writer,
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
            SharedMemory::<MP::EO>::new_lined(out_smem_num_lines, out_smem_line_size);
        let absolute_partition_position = partition_row * num_partitions_n + partition_col;
        let slice_start = num_tile_lines * absolute_partition_position;
        let mut smem_slice = out_smem.slice_mut(slice_start, slice_start + num_tile_lines);

        let m_offset = m_iterations * partition_row;
        let n_offset = n_iterations * partition_col;

        let mut m_iter = comptime![0u32];

        // Iterate over each tile in the partition
        #[unroll]
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..comptime![m_iterations] {
            let mut n_iter = comptime![0u32];

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..comptime![n_iterations] {
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
                Self::Writer::write::<G>(
                    out,
                    smem_slice.to_slice(),
                    m_offset + m_iter,
                    n_offset + n_iter,
                    global_config,
                );

                comptime![n_iter += 1];
            }
            comptime![m_iter += 1];
        }
    }

    fn init_writer(
        tensor: VirtualTensor<MP::EO, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
    ) -> Self::Writer {
        SP::init_writer(tensor, x_offset, y_offset, batch_offset)
    }
}

/// Different ways to schedule partition indices.
pub enum PartitionScheduleScheme {
    /// Apply offsets per plane to reduce shared memory conflicts (current scheme).
    Offset,
    /// Simple row-major mapping; no offsets, just global tiles in order.
    Naive,
}

/// Schedules global indices for M, N, and K axes in a partitioned matmul.
/// Each axis has its own `AxisScheduler`.
#[derive(CubeType)]
pub struct PartitionScheduler {
    pub m: AxisScheduler,
    pub n: AxisScheduler,
    pub k: AxisScheduler,
}

#[cube]
impl PartitionScheduler {
    /// Creates a scheduler for a partition at (partition_index_m, partition_index_n).
    /// Computes offsets so multiple partitions iterate over different tiles and reduce shared memory conflicts.
    pub fn new(
        partition_index_m: u32,
        partition_index_n: u32,
        #[comptime] partition_size: PartitionSize,
        #[comptime] partition_schedule_scheme: PartitionScheduleScheme,
    ) -> PartitionScheduler {
        match partition_schedule_scheme {
            PartitionScheduleScheme::Offset => {
                // M-axis rotation: ensures partitions in the same row start at different M tiles.
                let m_offset = (partition_index_n / partition_size.k()) % partition_size.m();

                // N-axis rotation: ensures partitions in the same column start at different N tiles.
                let n_offset = (partition_index_m / partition_size.k()) % partition_size.n();

                // K-axis rotation: simple offset; same diagonal can share K safely.
                let k_offset = (partition_index_m + partition_index_n) % partition_size.k();

                PartitionScheduler {
                    m: AxisScheduler::new_Offset(OffsetAxisScheduler::new(
                        m_offset,
                        partition_index_m,
                        partition_size.m(),
                    )),
                    n: AxisScheduler::new_Offset(OffsetAxisScheduler::new(
                        n_offset,
                        partition_index_n,
                        partition_size.n(),
                    )),
                    k: AxisScheduler::new_Offset(OffsetAxisScheduler::new(
                        k_offset,
                        0u32,
                        partition_size.k(),
                    )),
                }
            }
            PartitionScheduleScheme::Naive => PartitionScheduler {
                m: AxisScheduler::new_Naive(NaiveAxisScheduler::new(
                    partition_index_m,
                    partition_size.m(),
                )),
                n: AxisScheduler::new_Naive(NaiveAxisScheduler::new(
                    partition_index_n,
                    partition_size.n(),
                )),
                k: AxisScheduler::new_Naive(NaiveAxisScheduler::new(0u32, partition_size.k())),
            },
        }
    }

    /// Maps a local M index to a global index.
    pub fn map_m(&self, i: u32) -> u32 {
        self.m.map(i)
    }

    /// Maps a local N index to a global index.
    pub fn map_n(&self, i: u32) -> u32 {
        self.n.map(i)
    }

    /// Maps a local K index to a global index.
    pub fn map_k(&self, i: u32) -> u32 {
        self.k.map(i)
    }
}

#[derive(CubeType)]
#[allow(unused)]
pub enum AxisScheduler {
    Offset(OffsetAxisScheduler),
    Naive(NaiveAxisScheduler),
}

#[cube]
impl AxisScheduler {
    pub fn map(&self, i: u32) -> u32 {
        match self {
            AxisScheduler::Offset(offset_axis_scheduler) => offset_axis_scheduler.map(i),
            AxisScheduler::Naive(naive_axis_scheduler) => naive_axis_scheduler.map(i),
        }
    }
}

/// Schedules index mapping for one axis in a partitioned loop.
/// Computes a global index combining an intra-partition rotation (`inner_offset`)
/// and a partition-level shift (`outer_offset`), wrapping around within the partition.
#[derive(CubeType)]
pub struct OffsetAxisScheduler {
    /// Rotation inside this partition.
    inner_offset: u32,
    /// Starting index in the global axis, skipping previous partitions.
    outer_offset: u32,
    /// Number of tiles in this partition (compile-time constant).
    #[cube(comptime)]
    len: u32,
}

#[cube]
impl OffsetAxisScheduler {
    /// Creates a new `AxisScheduler`.
    ///
    /// # Arguments
    /// - `inner_offset`: rotation inside this partition.
    /// - `partition_index`: index of this partition along the axis.
    /// - `len`: number of tiles in this partition.
    pub fn new(
        inner_offset: u32,
        partition_index: u32,
        #[comptime] len: u32,
    ) -> OffsetAxisScheduler {
        let outer_offset = partition_index * len;
        OffsetAxisScheduler {
            inner_offset,
            outer_offset,
            len,
        }
    }

    /// Maps a local index `i` to the global index.
    /// Combines rotation (`inner_offset`) and global shift (`outer_offset`).
    pub fn map(&self, i: u32) -> u32 {
        let relative = (i + self.inner_offset) % self.len;
        relative + self.outer_offset
    }
}

#[derive(CubeType)]
pub struct NaiveAxisScheduler {
    outer_offset: u32,
}

#[cube]
impl NaiveAxisScheduler {
    pub fn new(partition_index: u32, #[comptime] len: u32) -> NaiveAxisScheduler {
        let outer_offset = partition_index * len;
        NaiveAxisScheduler { outer_offset }
    }

    pub fn map(&self, i: u32) -> u32 {
        i + self.outer_offset
    }
}
