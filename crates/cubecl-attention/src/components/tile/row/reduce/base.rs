use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::tile::PlaneLayout;
use crate::components::tile::RowMax;
use crate::components::tile::RowSum;
use crate::components::tile::RowWise;
use crate::components::tile::dummy::AttentionMatmulConfig;

#[cube]
pub fn row_sum<E: Float, PL: PlaneLayout<E>, R: Reducer, TC: AttentionMatmulConfig>(
    vals: &mut RowWise<E>,
    data: &PL,
    #[comptime] config: TC,
) {
    vals.copy_from(&RowWise::new_zero(vals.num_rows));
    R::reduce::<E, PL, RowSum, TC>(vals, data, config)
}

#[cube]
pub fn row_max<E: Float, PL: PlaneLayout<E>, R: Reducer, TC: AttentionMatmulConfig>(
    vals: &mut RowWise<E>,
    base: &RowWise<E>,
    data: &PL,
    #[comptime] config: TC,
) {
    vals.copy_from(base);
    R::reduce::<E, PL, RowMax, TC>(vals, data, config)
}

#[cube]
pub trait Reducer: CubeType {
    fn reduce<E: Float, PL: PlaneLayout<E>, RO: ReduceOp<E>, TC: AttentionMatmulConfig>(
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
