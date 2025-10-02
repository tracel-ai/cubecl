use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::AttentionPrecision;
use crate::components::TileMask;
use crate::components::attention_types::*;
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
    type PlaneLayout: PlaneLayout<E = SM<AP>>;
    type RowWise: RowWise<E = SM<AP>>;

    fn init_state(#[comptime] num_rows: u32) -> RunningState<Self::RowWise>;
    fn init_placeholder(#[comptime] num_rows: u32) -> Self::RowWise;

    fn zero(&mut self);

    fn scale_and_mask(&mut self, scale: SM<AP>, mask: TileMask);

    fn row_max(&self, placeholder: &mut Self::RowWise, base: &Self::RowWise);

    /// Converts scores → probabilities, updates running state,
    /// and returns the factor needed to scale the accumulator
    fn to_prob(
        &mut self,
        state: &mut RunningState<Self::RowWise>,
        max: &Self::RowWise,
        placeholder: &mut Self::RowWise,
    ) -> Self::RowWise;
}

#[cube]
pub trait AccumulatorTile<AP: AttentionPrecision, RW: RowWise>: CubeType {
    fn scale_mul(&mut self, scale: &RW);
    fn scale_div(&mut self, scale: &RW);
}
