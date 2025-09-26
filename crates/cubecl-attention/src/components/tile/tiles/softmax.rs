use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::AttentionPrecision;
use crate::components::TileMask;
use crate::components::attention_types::*;
use crate::components::tile::{RowWise, RunningState};

#[cube]
pub trait SoftmaxTile<AP: AttentionPrecision>: CubeType {
    fn init_state() -> RunningState<SM<AP>>;

    fn zero(&mut self);

    fn scale_and_mask(&mut self, scale: SM<AP>, mask: TileMask);

    fn row_max(&self, base: RowWise<SM<AP>>) -> RowWise<SM<AP>>;

    /// Converts scores → probabilities, updates running state,
    /// and returns the factor needed to scale the accumulator
    fn to_prob(
        &mut self,
        state: &mut RunningState<SM<AP>>,
        max: &RowWise<SM<AP>>,
    ) -> RowWise<ACC<AP>>;
}
