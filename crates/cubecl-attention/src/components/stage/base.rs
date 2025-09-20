use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::stage::StageReaderFamily;
use cubecl_std::CubeOption;
use cubecl_std::tensor::{View, layout::Coords2d};
use std::{fmt::Debug, hash::Hash};

use crate::components::AttentionTilingScheme;
use crate::components::global::dummy::QueryLoader;
use crate::components::stage::dummy::{AttentionStageMemoryConfig, StageState};
use crate::components::{
    AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError, AvailableLineSizes,
    global::GlobalAttentionConfig,
    tile::{AttentionTilingLayout, dummy::FlashMatmulConfig},
};

/// A family of [TileAttention] implementations that operate with any [precision](AttentionPrecision).
pub trait StageAttentionFamily: Send + Sync + 'static {
    /// The specific [TileAttention] implementation associated with this family.
    type Attention<AP: AttentionPrecision>: StageAttention<
            AP,
            Config = Self::Config,
            KeyReader = <Self::KeyReader as StageReaderFamily>::Reader<
                AP::ES,
                AttentionTilingLayout,
            >,
            ValueReader = <Self::ValueReader as StageReaderFamily>::Reader<
                AP::ES,
                AttentionTilingLayout,
            >,
        >;

    /// The configuration type associated with this Attention family.
    type Config: StageAttentionConfig;

    type KeyReader: StageReaderFamily;
    type ValueReader: StageReaderFamily;

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
    type KeyReader: CubeType;
    type ValueReader: CubeType;

    /// The configuration type associated with this Attention.
    type Config: StageAttentionConfig;

    type State: CubeType;

    type Query: CubeType;
    type KeyValue: CubeType;
    type Score: CubeType;
    type Accumulator: CubeType;

    type Writer: CubeType;

    fn init_state(#[comptime] config: Self::Config) -> StageState<AP>;

    fn execute(
        key_reader: &Self::KeyReader,
        value_reader: &Self::ValueReader,
        query: &Self::Query,
        key_value: &mut Self::KeyValue,
        score: &mut Self::Score,
        accumulator: &mut Self::Accumulator,
        prev_state: &mut StageState<AP>,
        out_of_bound_mask: CubeOption<(u32, u32)>,
        #[comptime] config: Self::Config,
    );

    fn rescale(
        acc: &mut Self::Accumulator,
        state: StageState<AP>,
        #[comptime] config: Self::Config,
    );

    fn write<G: GlobalAttentionConfig>(
        acc: &Self::Accumulator,
        writer: &mut Self::Writer,
        #[comptime] tile_config: Self::Config,
        #[comptime] global_config: G,
    );

    fn init_writer(tensor: View<Line<AP::EO>, Coords2d, ReadWrite>) -> Self::Writer;

    fn init_fragments(
        query_loader: QueryLoader<AP>,
        #[comptime] config: Self::Config,
    ) -> (Self::Query, Self::KeyValue, Self::Score, Self::Accumulator);
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
}
