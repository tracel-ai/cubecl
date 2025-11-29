use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::ComputeResources;
use cubecl_matmul::components::tile::StridedTile;

use crate::components::tile::{
    FragmentAccumulator, FragmentLayout, FragmentMask, FragmentSoftmax, RowwiseFormat,
};
use crate::components::{AttentionElems, attention_types::*};
use crate::components::{
    AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError, AttentionTileSize, AvailableLineSizes, InvalidConfigError,
};
use std::fmt::Debug;
use std::hash::Hash;

/// Logits below this are considered masked (effectively -inf)
/// Value chosen to fit within f16 range (~-65,504 max)
pub(crate) const LOGIT_MASKED: f32 = -6e4;

/// Any value smaller than this is considered numerically zero
/// (used for fully-masked rows or tiny contributions)
/// Value chosen to be above f16 smallest normal (~6.1e-5)
pub(crate) const FULLY_MASKED_ROW_THRESHOLD: f32 = 1e-4;

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

    fn load_query<E: Numeric>(tile: &StridedTile<E>, fragment: &mut Self::Query);

    fn load_key_transposed<E: Float>(
        tile: &StridedTile<E>,
        fragment: &mut Self::KeyValue,
        #[comptime] config: Self::Config,
    );
    fn load_value<E: Float>(
        tile: &StridedTile<E>,
        fragment: &mut Self::KeyValue,
        #[comptime] config: Self::Config,
    );
    fn load_mask<E: Numeric>(
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
    fn setup<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &AttentionProblem,
        selection: &AttentionSelection,
        line_sizes: &AttentionLineSizes,
        num_planes: u32,
        dtypes: &AttentionElems,
    ) -> Result<Self::Config, AttentionSetupError>;

    /// Filters out line sizes that are incompatible with this matmul family.
    ///
    /// By default, returns the input unchanged.
    fn filter_line_sizes(available_line_sizes: AvailableLineSizes) -> AvailableLineSizes {
        available_line_sizes
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct SharedTileAttentionConfig {
    pub plane_dim: u32,
    pub num_planes: u32,
    pub attention_tile_size: AttentionTileSize,
    pub causal_mask: bool,
    pub materialized_mask: bool,
}
