use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::AttentionPrecision;
use crate::components::attention_types::*;
use crate::components::tile::dummy::AttentionMatmulConfig;
use crate::components::tile::{FragmentOps, RowWise, RunningState};
use cubecl_matmul::components::tile::StridedTile;
use cubecl_std::CubeOption;
use cubecl_std::tensor::layout::Coords2d;

#[cube]
/// Query input to the Tile Attention
pub trait QueryTile<AP: AttentionPrecision>: CubeType {
    /// Loads the query data into the fragment
    fn update(&mut self, tile: &StridedTile<QG<AP>>);
}

#[cube]
/// Key and Value inputs to the Tile Attention
///
/// Key and Value share the same trait because they may
/// be the same reused underlying fragment
pub trait KeyValueTile<E: Float>: CubeType {
    /// The underlying fragment for key inputs
    type KeyFragment: CubeType;
    /// The underlying fragment for value inputs
    type ValueFragment: CubeType;

    /// Get the underlying key as readable
    fn key(&self) -> &Self::KeyFragment;
    /// Get the underlying key as writable
    fn key_mut(&mut self) -> &mut Self::KeyFragment;

    /// Get the underlying value as readable
    fn value(&self) -> &Self::ValueFragment;
    /// Get the underlying value as writable
    fn value_mut(&mut self) -> &mut Self::ValueFragment;
}

#[cube]
/// Softmax tile for the Tile Attention
///
/// This tile is neither an input nor an output,
/// but the intermediate step where the softmax part of attention happens
pub trait SoftmaxTile<AP: AttentionPrecision>: CubeType {
    /// The underlying fragment, for which operations must be defined
    type Fragment: FragmentOps<SM<AP>>;

    /// Init the running state used in softmax
    fn init_state(#[comptime] num_rows: u32) -> RunningState<SM<AP>>;

    /// Scale the tile by a constant factor and apply the mask
    fn scale_and_mask<M: MaskTile>(this: &mut Self, scale: SM<AP>, mask: &M);

    /// Compute the max of each row, starting with base
    /// as first element of the reduction, and storing result in placeholder
    fn row_max<TC: AttentionMatmulConfig>(
        &self,
        placeholder: &mut RowWise<SM<AP>>,
        base: &RowWise<SM<AP>>,
        #[comptime] config: TC,
    );

    /// Converts scores into (unnormalized) probabilities, updates running state,
    /// and returns the factor needed to scale the accumulator
    fn to_prob<TC: AttentionMatmulConfig>(
        &mut self,
        state: &mut RunningState<SM<AP>>,
        max: &RowWise<SM<AP>>,
        placeholder: &mut RowWise<SM<AP>>,
        #[comptime] config: TC,
    ) -> RowWise<SM<AP>>;
}

#[cube]
/// Accumulator tile for Tile Attention
pub trait AccumulatorTile<AP: AttentionPrecision>: CubeType {
    /// Multiplies each row by a scale
    fn scale_mul(&mut self, scale: &RowWise<SM<AP>>);

    /// Divides each row by a scale
    fn scale_div(&mut self, scale: &RowWise<SM<AP>>);
}

#[cube]
/// Mask tile for Tile Attention
/// It is an additive mask, which means the result of apply should be added, not multiplied
pub trait MaskTile: CubeType {
    /// The underlying fragment
    type Fragment: CubeType;
    /// Data type representing the boolean
    type MaskPrecision: Numeric;

    /// Returns -infinity if masked at local_pos, or zero if not
    fn apply<E: Float>(this: &Self, local_pos: Coords2d) -> E;

    /// Loads the mask data into the fragment, if a tile is given, otherwise only
    /// updates the logical mask
    fn update(&mut self, new_origin: Coords2d, tile: CubeOption<StridedTile<Self::MaskPrecision>>);
}
