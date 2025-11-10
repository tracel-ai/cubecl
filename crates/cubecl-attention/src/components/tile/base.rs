use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::ComputeResources;
use cubecl_matmul::components::tile::StridedTile;

use crate::components::attention_types::*;
use crate::components::tile::{
    FragmentAccumulator, FragmentLayout, FragmentMask, FragmentSoftmax, RowwiseFormat,
};
use crate::components::{
    AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError, AttentionTileSize, AvailableLineSizes, InvalidConfigError,
};
use std::fmt::Debug;
use std::hash::Hash;

#[cube]
pub trait TileAttention<AP: AttentionPrecision>: Send + Sync + 'static {
    type Config: TileAttentionConfig;
    type Query: CubeType;
    type KeyValue: CubeType;
    type Mask: FragmentMask<Layout = Self::FragmentLayout>;

    type Softmax: FragmentSoftmax<SM<AP>, Layout = Self::FragmentLayout, SoftmaxRowFormat = Self::SoftmaxRow>;
    type SoftmaxRow: RowwiseFormat<SM<AP>, Layout = Self::FragmentLayout>;

    type Accumulator: FragmentAccumulator<ACC<AP>>;
    type FragmentLayout: FragmentLayout;

    fn softmax_layout(#[comptime] config: Self::Config) -> Self::FragmentLayout;

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

    fn allocate_query(#[comptime] config: Self::Config) -> Self::Query;
    fn allocate_mask(#[comptime] config: Self::Config) -> Self::Mask;

    fn allocate_key(#[comptime] config: Self::Config) -> Self::KeyValue;
    fn allocate_value(#[comptime] config: Self::Config) -> Self::KeyValue;
    fn allocate_key_value(#[comptime] config: Self::Config) -> Self::KeyValue;

    fn allocate_softmax(#[comptime] config: Self::Config) -> Self::Softmax;
    fn allocate_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator;

    fn fill_query<E: Numeric>(tile: &StridedTile<E>, fragment: &mut Self::Query);

    fn fill_key_transposed<E: Float>(
        tile: &StridedTile<E>,
        fragment: &mut Self::KeyValue,
        #[comptime] config: Self::Config,
    );
    fn fill_value<E: Float>(
        tile: &StridedTile<E>,
        fragment: &mut Self::KeyValue,
        #[comptime] config: Self::Config,
    );
    fn fill_mask<E: Numeric>(
        tile: &StridedTile<E>,
        fragment: &mut Self::Mask,
        #[comptime] config: Self::Config,
    );

    fn write_results<E: Float>(
        out: &Self::Accumulator,
        slice: &mut SliceMut<Line<E>>,
        #[comptime] config: Self::Config,
    );
}

/// Configuration for the Tile Attention level
pub trait TileAttentionConfig:
    Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static
{
    fn plane_dim(&self) -> u32;
    fn num_planes(&self) -> u32;
    fn attention_tile_size(&self) -> AttentionTileSize;
    fn num_rows_per_unit(&self) -> u32;
    fn causal_mask(&self) -> bool;
    fn materialized_mask(&self) -> bool;
}

pub trait TileAttentionFamily: Send + Sync + 'static {
    /// The specific [TileMatmul] implementation associated with this family.
    type TileAttention<AP: AttentionPrecision>: TileAttention<AP, Config = Self::Config>;

    /// The configuration type associated with this matmul family.
    type Config: TileAttentionConfig;

    /// Returns whether this tile matmul requires specialized hardware accelerators (e.g., tensor cores).
    fn requires_accelerator() -> bool;

    /// Returns the compute resources required to run this tile matmul.
    fn computation_resources() -> Result<ComputeResources, InvalidConfigError>;

    /// Constructs the configuration based on the matmul problem, selection, and line sizes.
    ///
    /// This function may return an error if the configuration cannot be supported on the current runtime.
    fn setup<AP: AttentionPrecision, R: Runtime>(
        client: &ComputeClient<R::Server>,
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
