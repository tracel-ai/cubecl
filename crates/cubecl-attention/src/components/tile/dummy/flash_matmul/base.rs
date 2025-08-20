use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::tile::{Tile, TileConfig};
use cubecl_matmul::components::{ComputeResources, TileSize};

use crate::components::{
    AttentionLineSizes, AttentionProblem, AttentionSelection, AttentionSetupError,
    AvailableLineSizes, FlashIdent, InvalidConfigError,
};
use std::fmt::Debug;
use std::hash::Hash;

pub trait FlashPrecision: Send + Sync + Copy + 'static {
    type Q: Float;
    type KV: Float;
    type SP: Float;
    type A: Float;
}

#[cube]
pub trait FlashMatmul<FP: FlashPrecision>: Send + Sync + 'static {
    type Config: FlashMatmulConfig;
    type Query: CubeType;
    type KeyValue: CubeType;
    type ScoreProb: CubeType + Copy;
    type Accumulator: CubeType;

    fn score_matmul(lhs: &Self::Query, rhs: &Self::KeyValue, out: &mut Self::ScoreProb);

    fn value_matmul(lhs: &Self::ScoreProb, rhs: &Self::KeyValue, out: &mut Self::Accumulator);

    fn allocate_fill_query<EI: Numeric>(
        tile: &Tile<EI>,
        #[comptime] config: Self::Config,
    ) -> Self::Query;

    fn allocate_key_value(#[comptime] config: Self::Config) -> Self::KeyValue;

    fn fill_rhs<E: Numeric>(
        tile: &Tile<E>,
        rhs: &mut Self::KeyValue,
        #[comptime] config: Self::Config,
    );

    fn allocate_score_prob(#[comptime] config: Self::Config) -> Self::ScoreProb;
    fn zero_score_prob(score_prob: &mut Self::ScoreProb);

    fn allocate_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator;
    fn zero_accumulator(acc: &mut Self::Accumulator);

    fn write_results<E: Numeric>(
        out: &Self::Accumulator,
        slice: &mut SliceMut<Line<E>>,
        #[comptime] config: Self::Config,
    );

    // These methods should be deletable when we have proper control over fragments
    fn tmp_fill_accumulator(
        tile: &Tile<FP::A>,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    );
    fn tmp_fill_prob(
        tile: &Tile<FP::SP>,
        prob: &mut Self::ScoreProb,
        #[comptime] config: Self::Config,
    );
    fn tmp_write_score_prob<E: Numeric>(
        out: &Self::ScoreProb,
        slice: &mut SliceMut<Line<E>>,
        #[comptime] config: Self::Config,
    );
}

/// Configuration for the Tile Attention level
pub trait FlashMatmulConfig:
    Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static
{
    type ScoreConfig: TileConfig;
    type ValueConfig: TileConfig;

    fn score_config(&self) -> Self::ScoreConfig;
    fn value_config(&self) -> Self::ValueConfig;
    fn plane_dim(&self) -> u32;
    fn num_planes(&self) -> u32;
    fn rows_per_plane(&self) -> u32;
    fn reuse_key_value(&self) -> bool;
    fn stage_line_size(&self, ident: FlashIdent) -> u32;
    fn tile_size(&self) -> TileSize;
}

pub trait FlashMatmulFamily: Send + Sync + 'static {
    /// The specific [TileMatmul] implementation associated with this family.
    type Matmul<F: FlashPrecision>: FlashMatmul<F, Config = Self::Config>;

    /// The configuration type associated with this matmul family.
    type Config: FlashMatmulConfig;

    /// Returns whether this tile matmul requires specialized hardware accelerators (e.g., tensor cores).
    fn requires_accelerator() -> bool;

    /// Returns the compute resources required to run this tile matmul.
    fn computation_resources() -> Result<ComputeResources, InvalidConfigError>;

    /// Constructs the configuration based on the matmul problem, selection, and line sizes.
    ///
    /// This function may return an error if the configuration cannot be supported on the current runtime.
    fn setup<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &AttentionProblem,
        selection: &AttentionSelection,
        line_sizes: &AttentionLineSizes,
    ) -> Result<Self::Config, AttentionSetupError>;

    /// Filters out line sizes that are incompatible with this matmul family.
    ///
    /// By default, returns the input unchanged.
    fn filter_line_sizes(available_line_sizes: AvailableLineSizes) -> AvailableLineSizes {
        available_line_sizes
    }
}
