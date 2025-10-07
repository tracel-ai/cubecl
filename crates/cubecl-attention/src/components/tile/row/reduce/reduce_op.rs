use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::tile::ReduceOp;
use crate::components::tile::RowWise;
use crate::components::tile::{PlaneLayout, PlaneLayoutExpand};

#[derive(CubeType)]
pub struct RowMax {}

#[derive(CubeType)]
pub struct RowSum {}

#[cube]
impl<E: Float> ReduceOp<E> for RowMax {
    fn reduce_local<PL: PlaneLayout<E>>(data: &PL) -> RowWise<E> {
        data.rowwise_max()
    }

    fn reduce_local_store<PL: PlaneLayout<E>>(data: &PL, acc: &mut RowWise<E>) {
        acc.max_inplace(&Self::reduce_local::<PL>(data))
    }

    fn reduce_step_rowwise(acc: &mut RowWise<E>, elem: &RowWise<E>, mask: bool) {
        let mut masked = RowWise::new_filled(elem.num_rows, E::cast_from(mask) * E::min_value());
        masked.add_inplace(elem);

        acc.max_inplace(&masked)
    }

    fn reduce_step_scalar(a: E, b: E) -> E {
        Max::max(a, b)
    }
}

#[cube]
impl<E: Float> ReduceOp<E> for RowSum {
    fn reduce_local<PL: PlaneLayout<E>>(data: &PL) -> RowWise<E> {
        data.rowwise_sum()
    }

    fn reduce_local_store<PL: PlaneLayout<E>>(data: &PL, acc: &mut RowWise<E>) {
        acc.add_inplace(&Self::reduce_local::<PL>(data))
    }

    fn reduce_step_rowwise(acc: &mut RowWise<E>, elem: &RowWise<E>, mask: bool) {
        let mut masked = RowWise::new_filled(elem.num_rows, E::cast_from(!mask));
        masked.mul_inplace(elem);

        acc.add_inplace(&masked)
    }

    fn reduce_step_scalar(a: E, b: E) -> E {
        a + b
    }
}
