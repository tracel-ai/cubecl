use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::{
    stage::ReduceOp,
    tile::{RowWise, RowwiseFormat, RowwiseFormatExpand},
};

#[derive(CubeType)]
/// Max reduction operation
pub struct RowMax {}

#[derive(CubeType)]
/// Sum reduction operation
pub struct RowSum {}

#[cube]
impl<E: Float> ReduceOp<E> for RowMax {
    fn reduce_local<F: RowwiseFormat<E>>(data: &F) -> RowWise<E> {
        data.rowwise_max()
    }

    fn reduce_local_accumulate<F: RowwiseFormat<E>>(data: &F, acc: &mut RowWise<E>) {
        acc.max_inplace(&Self::reduce_local::<F>(data))
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
    fn reduce_local<F: RowwiseFormat<E>>(data: &F) -> RowWise<E> {
        data.rowwise_sum()
    }

    fn reduce_local_accumulate<F: RowwiseFormat<E>>(data: &F, acc: &mut RowWise<E>) {
        acc.add_inplace(&Self::reduce_local::<F>(data))
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
