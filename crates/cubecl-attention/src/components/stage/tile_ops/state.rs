use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::tile::RowWise;

#[derive(CubeType)]
/// Flash Attention's running state, per row
pub struct RunningState<E: Float> {
    m: RowWise<E>,
    l: RowWise<E>,
}

#[cube]
impl<E: Float> RunningState<E> {
    /// Init the state with neutral values
    pub fn init(#[comptime] num_rows: u32) -> RunningState<E> {
        RunningState::<E> {
            m: RowWise::new_min_value(num_rows),
            l: RowWise::new_zero(num_rows),
        }
    }

    /// Update the state for next iteration
    pub fn update(&mut self, new_m: &RowWise<E>, new_l: &RowWise<E>) {
        RowWise::copy_from(&mut self.m, new_m);
        RowWise::copy_from(&mut self.l, new_l);
    }

    /// Get the running m
    pub fn m(&self) -> &RowWise<E> {
        &self.m
    }

    /// Get the running l
    pub fn l(&self) -> &RowWise<E> {
        &self.l
    }
}
