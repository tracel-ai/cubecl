use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::tile::RowWise;

#[derive(CubeType)]
pub struct RunningState<E: Float> {
    pub m: RowWise<E>,
    pub l: RowWise<E>,
}

#[cube]
impl<E: Float> RunningState<E> {
    pub fn init(#[comptime] num_rows: u32) -> RunningState<E> {
        RunningState::<E> {
            m: RowWise::new_min_value(num_rows),
            l: RowWise::new_zero(num_rows),
        }
    }

    pub fn update(&mut self, new_m: &RowWise<E>, new_l: &RowWise<E>) {
        RowWise::copy_from(&mut self.m, new_m);
        RowWise::copy_from(&mut self.l, new_l);
    }
}

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
