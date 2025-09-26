use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::{global::memory::GlobalMemoryConfig, stage::StageFamily};
use cubecl_std::tensor::{View, layout::Coords2d};
use std::{fmt::Debug, hash::Hash};

use crate::components::{AttentionIdent, StageMask};
use crate::components::stage::dummy::AttentionStageMemoryConfig;
use crate::components::{
    AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError, AvailableLineSizes,
    global::GlobalAttentionConfig,
    tile::{AttentionTilingLayout, dummy::FlashMatmulConfig},
};
use crate::components::{AttentionTilingScheme, global::dummy::QueryReader};

/// A family of [TileAttention] implementations that operate with any [precision](AttentionPrecision).
pub trait StageAttentionFamily: Send + Sync + 'static {
    /// The specific [TileAttention] implementation associated with this family.
    type Attention<AP: AttentionPrecision>: StageAttention<
            AP,
            Config = Self::Config,
            KeyStage = <Self::KeyStage as StageFamily>::Stage<AP::ES, AttentionTilingLayout>,
            ValueStage = <Self::ValueStage as StageFamily>::Stage<AP::ES, AttentionTilingLayout>,
        >;

    /// The configuration type associated with this Attention family.
    type Config: StageAttentionConfig;

    type KeyStage: StageFamily;
    type ValueStage: StageFamily;

    /// Constructs the configuration based on the Attention problem, selection, and line sizes.
    ///
    /// This function may return an error if the configuration cannot be supported on the current runtime.
    fn setup<AP: AttentionPrecision, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &AttentionProblem,
        selection: &AttentionSelection,
        line_sizes: &AttentionLineSizes,
    ) -> Result<Self::Config, AttentionSetupError>;

    /// Filters out line sizes that are incompatible with this Attention family.
    ///
    /// By default, returns the input unchanged.
    fn filter_line_sizes(available_line_sizes: AvailableLineSizes) -> AvailableLineSizes {
        available_line_sizes
    }
}

#[cube]
pub trait StageAttention<AP: AttentionPrecision>: 'static + Send + Sync {
    type KeyStage: CubeType;
    type ValueStage: CubeType;

    /// The configuration type associated with this Attention.
    type Config: StageAttentionConfig;

    type State: CubeType;

    type QueryPartition: CubeType;
    type KeyValuePartition: CubeType;
    type SoftmaxPartition: CubeType;
    type AccumulatorPartition: CubeType;

    type Writer: CubeType;

    fn init_state(#[comptime] config: Self::Config) -> Self::State;

    fn execute(
        key_reader: &Self::KeyStage,
        value_reader: &Self::ValueStage,
        query: &Self::QueryPartition,
        key_value: &mut Self::KeyValuePartition,
        score: &mut Self::SoftmaxPartition,
        mask: StageMask,
        accumulator: &mut Self::AccumulatorPartition,
        prev_state: &mut Self::State,
        #[comptime] config: Self::Config,
    );

    fn rescale(
        acc: &mut Self::AccumulatorPartition,
        state: Self::State,
        #[comptime] config: Self::Config,
    );

    fn write<G: GlobalAttentionConfig>(
        acc: &Self::AccumulatorPartition,
        writer: &mut Self::Writer,
        #[comptime] tile_config: Self::Config,
        #[comptime] global_config: G,
    );

    fn init_writer(
        tensor: View<Line<AP::EO>, Coords2d, ReadWrite>,
        #[comptime] config: GlobalMemoryConfig,
    ) -> Self::Writer;

    fn init_partitions(
        query_loader: QueryReader<AP>,
        #[comptime] config: Self::Config,
    ) -> (
        Self::QueryPartition,
        Self::KeyValuePartition,
        Self::SoftmaxPartition,
        Self::AccumulatorPartition,
    );
}

/// Configuration for the Tile Attention level
pub trait StageAttentionConfig:
    Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static
{
    type FlashMatmulConfig: FlashMatmulConfig;

    fn plane_dim(&self) -> u32;
    fn num_planes(&self) -> u32;

    fn tile_config(&self) -> Self::FlashMatmulConfig;
    fn score_stage_memory_config(&self) -> AttentionStageMemoryConfig;
    fn value_stage_memory_config(&self) -> AttentionStageMemoryConfig;

    fn tiling_scheme(&self) -> AttentionTilingScheme;
    fn reuse_key_value(&self) -> bool;

    fn num_rows_per_unit(&self, ident: AttentionIdent) -> u32;
}
