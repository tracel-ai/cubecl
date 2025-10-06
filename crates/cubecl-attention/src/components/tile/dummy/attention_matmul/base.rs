use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::ComputeResources;
use cubecl_matmul::components::tile::StridedTile;

use crate::components::attention_types::*;
use crate::components::tile::PlaneLayout;
use crate::components::{
    AttentionIdent, AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError, AttentionTileSize, AvailableLineSizes, InvalidConfigError,
};
use std::fmt::Debug;
use std::hash::Hash;

#[cube]
pub trait AttentionMatmul<AP: AttentionPrecision>: Send + Sync + 'static {
    type Config: AttentionMatmulConfig;
    type Query: CubeType;
    type KeyValue: CubeType;
    type Softmax: PlaneLayout<SM<AP>>;
    type Accumulator: PlaneLayout<ACC<AP>>;

    fn score_matmul(
        lhs: &Self::Query,
        rhs: &Self::KeyValue,
        out: &mut Self::Softmax,
        #[comptime] config: Self::Config,
    );

    fn value_matmul(
        lhs: &Self::Softmax,
        rhs: &Self::KeyValue,
        out: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    );

    fn allocate_fill_query<EI: Float>(
        tile: &StridedTile<EI>,
        #[comptime] config: Self::Config,
    ) -> Self::Query;

    fn allocate_key(#[comptime] config: Self::Config) -> Self::KeyValue;
    fn allocate_value(#[comptime] config: Self::Config) -> Self::KeyValue;
    fn allocate_key_value(#[comptime] config: Self::Config) -> Self::KeyValue;

    fn fill_key_value<E: Float>(
        tile: &StridedTile<E>,
        rhs: &mut Self::KeyValue,
        #[comptime] config: Self::Config,
    );

    fn allocate_softmax(#[comptime] config: Self::Config) -> Self::Softmax;
    fn zero_softmax(softmax: &mut Self::Softmax, #[comptime] config: Self::Config);

    fn allocate_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator;
    fn zero_accumulator(acc: &mut Self::Accumulator);

    fn write_results<E: Float>(
        out: &Self::Accumulator,
        slice: &mut SliceMut<Line<E>>,
        #[comptime] config: Self::Config,
    );
}

/// Configuration for the Tile Attention level
pub trait AttentionMatmulConfig:
    Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static
{
    fn plane_dim(&self) -> u32;

    // TODO try to remove this
    fn num_planes(&self) -> u32;
    fn stage_line_size(&self, ident: AttentionIdent) -> u32;
    fn attention_tile_size(&self) -> AttentionTileSize;
    // If AP::EI != FP::Q
    fn cast_query(&self) -> bool;

    fn check_bounds(&self) -> bool;

    fn num_rows_per_unit(&self) -> u32;
}

pub trait AttentionMatmulFamily: Send + Sync + 'static {
    /// The specific [TileMatmul] implementation associated with this family.
    type Matmul<AP: AttentionPrecision>: AttentionMatmul<AP, Config = Self::Config>;

    /// The configuration type associated with this matmul family.
    type Config: AttentionMatmulConfig;

    /// Returns whether this tile matmul requires specialized hardware accelerators (e.g., tensor cores).
    fn requires_accelerator() -> bool;

    /// Returns the compute resources required to run this tile matmul.
    fn computation_resources() -> Result<ComputeResources, InvalidConfigError>;

    /// Constructs the configuration based on the matmul problem, selection, and line sizes.
    ///
    /// This function may return an error if the configuration cannot be supported on the current runtime.
    fn setup<AP: AttentionPrecision, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &AttentionProblem,
        selection: &AttentionSelection,
        line_sizes: &AttentionLineSizes,
        num_planes: u32,
    ) -> Result<Self::Config, AttentionSetupError>;

    /// Filters out line sizes that are incompatible with this matmul family.
    ///
    /// By default, returns the input unchanged.
    fn filter_line_sizes(available_line_sizes: AvailableLineSizes) -> AvailableLineSizes {
        available_line_sizes
    }
}
