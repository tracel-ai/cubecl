use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::tile::RowWise;
use crate::components::tile::dummy::AttentionMatmulConfig;
use crate::components::tile::{PlaneLayout, PlaneLayoutExpand};

#[cube]
pub fn row_sum<E: Float, PL: PlaneLayout<E>, PO: Reducer, TC: AttentionMatmulConfig>(
    vals: &mut RowWise<E>,
    data: &PL,
    #[comptime] config: TC,
) {
    vals.copy_from(&RowWise::new_zero(vals.num_rows));
    PO::row_op::<E, PL, RowSum, TC>(vals, data, config)
}

#[cube]
pub fn row_max<E: Float, PL: PlaneLayout<E>, PO: Reducer, TC: AttentionMatmulConfig>(
    vals: &mut RowWise<E>,
    base: &RowWise<E>,
    data: &PL,
    #[comptime] config: TC,
) {
    vals.copy_from(base);
    PO::row_op::<E, PL, RowMax, TC>(vals, data, config)
}

#[cube]
pub trait Reducer: CubeType {
    fn row_op<E: Float, PL: PlaneLayout<E>, RO: ReduceOp<E>, TC: AttentionMatmulConfig>(
        vals: &mut RowWise<E>,
        data: &PL,
        #[comptime] config: TC,
    );
}

#[cube]
pub trait ReduceOp<E: Float> {
    fn reduce_local<PL: PlaneLayout<E>>(data: &PL) -> RowWise<E>;
    fn reduce_local_store<PL: PlaneLayout<E>>(data: &PL, acc: &mut RowWise<E>);
    fn reduce_step_rowwise(acc: &mut RowWise<E>, elem: &RowWise<E>, mask: bool);
    fn reduce_step_scalar(a: E, b: E) -> E;
}

#[derive(CubeType)]
struct RowMax {}

#[derive(CubeType)]
struct RowSum {}

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
        masked.add_inplace(&elem);

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
        masked.mul_inplace(&elem);

        acc.add_inplace(&masked)
    }

    fn reduce_step_scalar(a: E, b: E) -> E {
        a + b
    }
}
