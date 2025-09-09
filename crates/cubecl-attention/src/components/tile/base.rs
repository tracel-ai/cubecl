use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::{
    stage::{ContiguousTilingLayout, RowMajorTilingOrder},
    tile::Tile,
};
use cubecl_std::tensor::{View, layout::Coords3d};

use crate::components::global::dummy::QueryRegisterReader;
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
    type Score: CubeType;
    type Accumulator: CubeType;

    fn init_state(#[comptime] config: Self::Config) -> Self::State;

    fn execute(
        key_tile: &Tile<AP::ES>,
        value_tile: &Tile<AP::ES>,
        query: &Self::Query,
        key_value: &mut Self::KeyValue,
        score: &mut Self::Score,
        accumulator: &mut Self::Accumulator,
        state: &mut Self::State,
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

    fn init_writer(tensor: View<Line<AP::EO>, Coords3d, ReadWrite>) -> Self::Writer;

    fn init_fragments(
        query_reader: QueryRegisterReader<AP::EI>,
        #[comptime] config: Self::Config,
    ) -> (Self::Query, Self::KeyValue, Self::Score, Self::Accumulator);
}
