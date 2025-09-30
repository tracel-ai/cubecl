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
            m: RowWise::new_filled(num_rows, E::min_value()),
            l: RowWise::new_filled(num_rows, E::from_int(0)),
        }
    }

    pub fn update(&mut self, new_m: &RowWise<E>, new_l: &RowWise<E>) {
        self.m.copy_from(new_m);
        self.l.copy_from(new_l);
    }
}
