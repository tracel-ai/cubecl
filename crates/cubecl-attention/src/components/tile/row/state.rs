use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::tile::RowWise;

#[derive(CubeType)]
pub struct RunningState<RW: RowWise> {
    pub m: RW,
    pub l: RW,
}

#[cube]
impl<RW: RowWise> RunningState<RW> {
    pub fn init(#[comptime] num_rows: u32) -> RunningState<RW> {
        RunningState::<RW> {
            m: RW::new_filled(num_rows, RW::E::min_value()),
            l: RW::new_filled(num_rows, RW::E::from_int(0)),
        }
    }

    pub fn update(&mut self, new_m: &RW, new_l: &RW) {
        RW::copy_from(&mut self.m, new_m);
        RW::copy_from(&mut self.l, new_l);
    }
}

#[derive(CubeType)]
pub struct RowStats<RW: RowWise> {
    pub score_max: RW,
    pub prob_sum: RW,
}

#[cube]
impl<RW: RowWise> RowStats<RW> {
    pub fn new(score_max: RW, prob_sum: RW) -> RowStats<RW> {
        RowStats::<RW> {
            score_max,
            prob_sum,
        }
    }
}
