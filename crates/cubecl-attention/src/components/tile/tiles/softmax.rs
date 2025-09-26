use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::TileMask;
use crate::components::tile::dummy::FlashPrecision;
use crate::components::tile::{RowWise, RunningState};

#[cube]
pub trait SoftmaxTile<FP: FlashPrecision>: CubeType {
    fn init_state() -> RunningState<FP::SP>;

    fn zero(&mut self);

    fn scale_and_mask(&mut self, scale: FP::SP, mask: TileMask);

    fn row_max(&self, base: RowWise<FP::SP>) -> RowWise<FP::SP>;

    /// Converts scores → probabilities, updates running state,
    /// and returns the factor needed to scale the accumulator
    fn to_prob(
        &mut self,
        state: &mut RunningState<FP::SP>,
        max: &RowWise<FP::SP>,
    ) -> RowWise<FP::A>;
}
