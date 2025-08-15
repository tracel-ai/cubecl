use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::{
    stage::{ContiguousTilingLayout, RowMajorTilingOrder},
    tile::{Tile, TileConfig, TileMatmul},
};
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};

use crate::components::global::dummy::QueryRegisterReader;
use crate::components::{
    AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError, AvailableLineSizes, global::GlobalAttentionConfig,
};
use std::{fmt::Debug, hash::Hash};

pub type AttentionTilingLayout = ContiguousTilingLayout<RowMajorTilingOrder>;

/// A family of [TileAttention] implementations that operate with any [precision](AttentionPrecision).
pub trait TileAttentionFamily: Send + Sync + 'static {
    /// The specific [TileAttention] implementation associated with this family.
    type Attention<AP: AttentionPrecision>: TileAttention<AP, Config = Self::Config>;

    /// The configuration type associated with this Attention family.
    type Config: TileAttentionConfig;

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
    type Config: TileAttentionConfig;

    type State: CubeType;

    type Query: CubeType;
    type KeyValue: CubeType;
    type ScoreProb: CubeType;
    type Accumulator: CubeType;

    fn init_state(#[comptime] config: Self::Config) -> Self::State;

    fn execute(
        key_tile: &Tile<AP::ES>,
        value_tile: &Tile<AP::ES>,
        query: &Self::Query,
        key_value: &mut Self::KeyValue,
        score_prob: &mut Self::ScoreProb,
        accumulator: &mut Self::Accumulator,
        prev_state: &mut Self::State,
        #[comptime] config: Self::Config,
    );

    fn rescale(
        acc: &mut Self::Accumulator,
        prev_state: Self::State,
        #[comptime] config: Self::Config,
    );

    fn write<G: GlobalAttentionConfig>(
        acc: &Self::Accumulator,
        writer: &mut Self::Writer,
        #[comptime] tile_config: Self::Config,
        #[comptime] global_config: G,
    );

    fn init_writer(tensor: VirtualTensor<AP::EO, ReadWrite>) -> Self::Writer;

    fn init_fragments(
        query_reader: QueryRegisterReader<AP>,
        #[comptime] config: Self::Config,
    ) -> (
        Self::Query,
        Self::KeyValue,
        Self::ScoreProb,
        Self::Accumulator,
    );
}

/// Configuration for the Tile Attention level
pub trait TileAttentionConfig:
    Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static
{
    type ScoreConfig: TileConfig;
    type ValueConfig: TileConfig;

    fn plane_dim(&self) -> u32;
    fn num_planes(&self) -> u32;
    fn rows_per_plane(&self) -> u32;

    fn score_config(&self) -> Self::ScoreConfig;
    fn value_config(&self) -> Self::ValueConfig;

    fn reuse_key_value(&self) -> bool;
}

pub trait ScoreMatmul<AP: AttentionPrecision>: TileMatmul<AP::ES, AP::ES, AP::EA> {}
impl<AP, T> ScoreMatmul<AP> for T
where
    AP: AttentionPrecision,
    T: TileMatmul<AP::ES, AP::ES, AP::EA>,
{
}

pub trait ValueMatmul<AP: AttentionPrecision>: TileMatmul<AP::EA, AP::ES, AP::EA> {}
impl<AP, T> ValueMatmul<AP> for T
where
    AP: AttentionPrecision,
    T: TileMatmul<AP::EA, AP::ES, AP::EA>,
{
}
