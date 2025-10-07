use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::AttentionPrecision;
use crate::components::TileMask;
use crate::components::attention_types::*;
use crate::components::tile::dummy::AttentionMatmulConfig;
use crate::components::tile::{PlaneLayout, RowWise, RunningState};

#[cube]
pub trait QueryTile<E: Float>: CubeType {}

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
    type PlaneLayout: PlaneLayout<SM<AP>>;

    fn init_state(#[comptime] num_rows: u32) -> RunningState<SM<AP>>;
    fn init_placeholder(#[comptime] num_rows: u32) -> RowWise<SM<AP>>;

    fn zero(&mut self);

    fn scale_and_mask(&mut self, scale: SM<AP>, mask: TileMask);

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
