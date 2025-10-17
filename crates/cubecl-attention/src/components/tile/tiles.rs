use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::AttentionPrecision;
use crate::components::attention_types::*;
use crate::components::tile::FragmentMask;
use crate::components::tile::dummy::AttentionMatmulConfig;
use crate::components::tile::{FragmentOps, RowWise, RunningState};
use cubecl_matmul::components::tile::StridedTile;
use cubecl_std::CubeOption;
use cubecl_std::tensor::layout::Coords2d;

#[cube]
pub trait QueryTile<AP: AttentionPrecision>: CubeType {
    type Fragment: CubeType;
    type Config: AttentionMatmulConfig;

    fn fragment_mut(&mut self) -> &mut Self::Fragment;
    fn update(&mut self, tile: StridedTile<QG<AP>>, #[comptime] config: Self::Config);
}

#[cube]
pub trait KeyValueTile<E: Float>: CubeType {
    type Key: CubeType;
    type Value: CubeType;

    fn key(&self) -> &Self::Key;
    fn key_mut(&mut self) -> &mut Self::Key;

    fn value(&self) -> &Self::Value;
    fn value_mut(&mut self) -> &mut Self::Value;
}

#[cube]
pub trait SoftmaxTile<AP: AttentionPrecision>: CubeType {
    type FragmentOps: FragmentOps<SM<AP>>;

    fn init_state(#[comptime] num_rows: u32) -> RunningState<SM<AP>>;
    fn init_placeholder(#[comptime] num_rows: u32) -> RowWise<SM<AP>>;

    fn zero(&mut self);

    fn scale_and_mask<M: MaskTile>(this: &mut Self, scale: SM<AP>, mask: &M);

    fn row_max<TC: AttentionMatmulConfig>(
        &self,
        placeholder: &mut RowWise<SM<AP>>,
        base: &RowWise<SM<AP>>,
        #[comptime] config: TC,
    );

    /// Converts scores → probabilities, updates running state,
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
pub trait AccumulatorTile<AP: AttentionPrecision>: CubeType {
    fn scale_mul(&mut self, scale: &RowWise<SM<AP>>);
    fn scale_div(&mut self, scale: &RowWise<SM<AP>>);
}

#[cube]
pub trait MaskTile: CubeType {
    type Fragment: CubeType;
    type MaskPrecision: Numeric;

    fn apply<E: Float>(this: &Self, local_pos: Coords2d) -> E;
    fn fragment_mut(&mut self) -> &mut Self::Fragment;
    fn update(&mut self, new_origin: Coords2d, tile: CubeOption<StridedTile<Self::MaskPrecision>>);
}
