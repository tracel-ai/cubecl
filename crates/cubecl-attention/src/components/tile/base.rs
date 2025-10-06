use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::{
    ComputeResources,
    stage::{ContiguousTilingLayout, RowMajorTilingOrder},
    tile::StridedTile,
};

use crate::components::{
    AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError, AvailableLineSizes,
    attention_types::*,
    tile::{KeyValueTile, QueryTile, RowWise, RunningState, dummy::AttentionMatmulConfig},
};
use crate::components::{InvalidConfigError, tile::AccumulatorTile};
use crate::components::{TileMask, tile::SoftmaxTile};

pub type AttentionTilingLayout = ContiguousTilingLayout<RowMajorTilingOrder>;

/// A family of [TileAttention] implementations that operate with any [precision](AttentionPrecision).
pub trait TileAttentionFamily: Send + Sync + 'static {
    /// The specific [TileAttention] implementation associated with this family.
    type Attention<AP: AttentionPrecision>: TileAttention<AP, Config = Self::Config>;

    /// The configuration type associated with this Attention family.
    type Config: AttentionMatmulConfig;

    /// Constructs the configuration based on the Attention problem, selection, and line sizes.
    ///
    /// This function may return an error if the configuration cannot be supported on the current runtime.
    fn setup<AP: AttentionPrecision, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &AttentionProblem,
        selection: &AttentionSelection,
        line_sizes: &AttentionLineSizes,
        num_planes: u32,
    ) -> Result<Self::Config, AttentionSetupError>;

    /// Filters out line sizes that are incompatible with this Attention family.
    ///
    /// By default, returns the input unchanged.
    fn filter_line_sizes(available_line_sizes: AvailableLineSizes) -> AvailableLineSizes {
        available_line_sizes
    }

    fn computation_resources() -> Result<ComputeResources, InvalidConfigError>;
}

#[cube]
pub trait TileAttention<AP: AttentionPrecision>: 'static + Send + Sync {
    /// The configuration type associated with this Attention.
    type Config: AttentionMatmulConfig;

    type QueryTile: QueryTile<QT<AP>>;
    type KeyValueTile: KeyValueTile<KVT<AP>>;
    type SoftmaxTile: SoftmaxTile<AP>;
    type AccumulatorTile: AccumulatorTile<AP>;

    fn rescale(
        acc: &mut Self::AccumulatorTile,
        prev_state: &RunningState<SM<AP>>,
        #[comptime] config: Self::Config,
    );

    fn write_results(
        tile: &mut StridedTile<OS<AP>, ReadWrite>,
        acc: &Self::AccumulatorTile,
        #[comptime] tile_config: Self::Config,
    );

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::AccumulatorTile;

    fn init_query(tile: &StridedTile<QG<AP>>, #[comptime] config: Self::Config) -> Self::QueryTile;

    fn init_key_value(#[comptime] config: Self::Config) -> Self::KeyValueTile;
    fn init_key(#[comptime] config: Self::Config) -> Self::KeyValueTile;
    fn init_value(#[comptime] config: Self::Config) -> Self::KeyValueTile;

    fn init_softmax(#[comptime] config: Self::Config) -> Self::SoftmaxTile;

    fn init_state(#[comptime] config: Self::Config) -> RunningState<SM<AP>>;

    fn fill_key<E: Float>(
        tile: &StridedTile<E>,
        rhs: &mut Self::KeyValueTile,
        #[comptime] config: Self::Config,
    );

    fn fill_value<E: Float>(
        tile: &StridedTile<E>,
        rhs: &mut Self::KeyValueTile,
        #[comptime] config: Self::Config,
    );

    fn zero_softmax(score: &mut Self::SoftmaxTile, #[comptime] config: Self::Config);

    fn accumulate_score(
        query: &Self::QueryTile,
        key_value: &Self::KeyValueTile,
        softmax: &mut Self::SoftmaxTile,
        #[comptime] config: Self::Config,
    );

    fn softmax(
        softmax: &mut Self::SoftmaxTile,
        mask: TileMask,
        state: &mut RunningState<SM<AP>>,
        max_placeholder: &mut RowWise<SM<AP>>,
        sum_placeholder: &mut RowWise<SM<AP>>,
        #[comptime] dk: u32,
        #[comptime] config: Self::Config,
    ) -> RowWise<SM<AP>>;

    fn accumulate_value(
        softmax: &Self::SoftmaxTile,
        key_value: &Self::KeyValueTile,
        accumulator: &mut Self::AccumulatorTile,
        scale: &RowWise<SM<AP>>,
        #[comptime] config: Self::Config,
    );

    fn init_max_placeholder(#[comptime] num_rows: u32) -> RowWise<SM<AP>>;
    fn init_sum_placeholder(#[comptime] num_rows: u32) -> RowWise<SM<AP>>;
}
