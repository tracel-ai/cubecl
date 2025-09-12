use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::{
    stage::{ContiguousTilingLayout, RowMajorTilingOrder},
    tile::Tile,
};
use cubecl_std::CubeOption;
use cubecl_std::tensor::{View, layout::Coords3d};

use crate::components::{
    AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError, AvailableLineSizes, global::GlobalAttentionConfig,
    tile::dummy::FlashMatmulConfig,
};

pub type AttentionTilingLayout = ContiguousTilingLayout<RowMajorTilingOrder>;

/// A family of [TileAttention] implementations that operate with any [precision](AttentionPrecision).
pub trait TileAttentionFamily: Send + Sync + 'static {
    /// The specific [TileAttention] implementation associated with this family.
    type Attention<AP: AttentionPrecision>: TileAttention<AP, Config = Self::Config>;

    /// The configuration type associated with this Attention family.
    type Config: FlashMatmulConfig;

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
pub trait TileAttention<AP: AttentionPrecision>: 'static + Send + Sync {
    type Writer: CubeType;

    /// The configuration type associated with this Attention.
    type Config: FlashMatmulConfig;

    type State: CubeType;

    type Query: CubeType;
    type KeyValue: CubeType;
    type ScoreProb: CubeType;
    type Accumulator: CubeType;
    type OutOfBoundMask: CubeType;

    fn init_state(#[comptime] config: Self::Config) -> Self::State;

    fn execute(
        key_tile: &Tile<AP::ES>,
        value_tile: &Tile<AP::ES>,
        query: &Self::Query,
        key_value: &mut Self::KeyValue,
        score: &mut Self::ScoreProb,
        accumulator: &mut Self::Accumulator,
        state: &mut Self::State,
        out_of_bound_mask: CubeOption<(u32, u32)>,
        #[comptime] config: Self::Config,
    );

    fn rescale(
        acc: &mut Self::Accumulator,
        prev_state: &Self::State,
        #[comptime] config: Self::Config,
    );

    fn write<G: GlobalAttentionConfig>(
        acc: &Self::Accumulator,
        writer: &mut Self::Writer,
        #[comptime] tile_config: Self::Config,
        #[comptime] global_config: G,
    );

    fn init_writer(q_offset: u32, tensor: View<Line<AP::EO>, Coords3d, ReadWrite>) -> Self::Writer;

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator;

    fn init_query(tile: &Tile<AP::EI>, #[comptime] config: Self::Config) -> Self::Query;

    fn init_key_value(#[comptime] config: Self::Config) -> Self::KeyValue;

    fn init_score(#[comptime] config: Self::Config) -> Self::ScoreProb;

    fn fill_key<E: Numeric>(
        tile: &Tile<E>,
        rhs: &mut Self::KeyValue,
        #[comptime] config: Self::Config,
    );

    fn fill_value<E: Numeric>(
        tile: &Tile<E>,
        rhs: &mut Self::KeyValue,
        #[comptime] config: Self::Config,
    );

    fn zero_score(score: &mut Self::ScoreProb, #[comptime] config: Self::Config);

    fn accumulate_score(
        query: &Self::Query,
        key_value: &Self::KeyValue,
        score_prob: &mut Self::ScoreProb,
        #[comptime] config: Self::Config,
    );

    fn score_to_prob(
        score_prob: &mut Self::ScoreProb,
        out_of_bound_mask: CubeOption<(u32, u32)>,
        state: &Self::State,
        #[comptime] config: Self::Config,
    ) -> RowStats<AP::EA>;

    fn accumulate_value(
        key_value: &Self::KeyValue,
        score_prob: &Self::ScoreProb,
        accumulator: &mut Self::Accumulator,
        row_stats: &RowStats<AP::EA>,
        state: &mut Self::State,
        #[comptime] config: Self::Config,
    );
}

#[derive(CubeType)]
pub struct RowStats<E: Numeric> {
    pub m: E,
    pub prob_row_sum: E,
}
