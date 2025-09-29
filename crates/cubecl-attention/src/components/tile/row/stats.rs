use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::tile::RowWise;

#[derive(CubeType)]
pub struct RowStats<E: Float> {
    pub score_max: RowWise<E>,
    pub prob_sum: RowWise<E>,
}

#[cube]
impl<E: Float> RowStats<E> {
    pub fn new(score_max: RowWise<E>, prob_sum: RowWise<E>) -> RowStats<E> {
        RowStats::<E> {
            score_max,
            prob_sum,
        }
    }
}
