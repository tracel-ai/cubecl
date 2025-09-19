use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::{
    stage::{ContiguousTilingLayout, RowMajorTilingOrder},
    tile::Tile,
};
use cubecl_std::CubeOption;

use crate::components::tile::dummy::RunningState;
use crate::components::{
    AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError, AvailableLineSizes, tile::dummy::FlashMatmulConfig,
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
    /// The configuration type associated with this Attention.
    type Config: FlashMatmulConfig;

    type Query: CubeType;
    type KeyValue: CubeType;
    type ScoreProb: CubeType;
    type Accumulator: CubeType;
    type OutOfBoundMask: CubeType;

    fn rescale(
        acc: &mut Self::Accumulator,
        prev_state: &RunningState<AP::EA>,
        #[comptime] config: Self::Config,
    );

    fn write_results(
        acc: &Self::Accumulator,
        slice: &mut SliceMut<Line<AP::EO>>,
        #[comptime] tile_config: Self::Config,
    );

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator;

    fn init_query(tile: &Tile<AP::EI>, #[comptime] config: Self::Config) -> Self::Query;

    fn init_key_value(#[comptime] config: Self::Config) -> Self::KeyValue;
    fn init_key(#[comptime] config: Self::Config) -> Self::KeyValue;
    fn init_value(#[comptime] config: Self::Config) -> Self::KeyValue;

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
        state: &RunningState<AP::EA>,
        #[comptime] dk: u32,
    ) -> RowStats<AP::EA>;

    fn update_state(
        state: &mut RunningState<AP::EA>,
        score_prob_row_stats: &RowStats<AP::EA>,
    ) -> AP::EA;

    fn accumulate_value(
        score_prob: &Self::ScoreProb,
        key_value: &Self::KeyValue,
        accumulator: &mut Self::Accumulator,
        scale: AP::EA,
        #[comptime] config: Self::Config,
    );
}

#[derive(CubeType)]
pub struct RowStats<E: Numeric> {
    pub m: E,
    pub prob_row_sum: E,
}
