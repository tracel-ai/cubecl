use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::tile::{RowElement, RowFormat};

#[derive(CubeType)]
pub struct RunningState<E: Float, RF: RowFormat> {
    pub m: RF::RowElement<E>,
    pub l: RF::RowElement<E>,
}

#[cube]
impl<E: Float, RF: RowFormat> RunningState<E, RF> {
    pub fn init() -> RunningState<E, RF> {
        RunningState::<E, RF> {
            m: RF::new_filled(E::from_int(-99999999999)),
            l: RF::new_filled(E::from_int(0)),
        }
    }

    pub fn update(&mut self, new_m: RF::RowElement<E>, new_l: RF::RowElement<E>) {
        <RF::RowElement<E> as RowElement<E>>::copy(&new_m, &mut self.m);
        <RF::RowElement<E> as RowElement<E>>::copy(&new_l, &mut self.l);
    }
}
