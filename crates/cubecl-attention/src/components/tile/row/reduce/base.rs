use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::fragment::AttentionMatmulConfig;
use crate::components::fragment::FragmentOps;
use crate::components::tile::RowMax;
use crate::components::tile::RowSum;
use crate::components::tile::RowWise;

#[cube]
/// Computes the sum of rows on a fragment, using the Reducer's strategy
pub fn row_sum<E: Float, F: FragmentOps<E>, R: Reducer, TC: AttentionMatmulConfig>(
    vals: &mut RowWise<E>,
    data: &F,
    #[comptime] config: TC,
) {
    vals.fill(E::from_int(0));
    R::reduce::<E, F, RowSum, TC>(vals, data, config)
}

#[cube]
/// Computes the max of rows on a fragment, using the Reducer's strategy
/// Starts max at base
pub fn row_max<E: Float, F: FragmentOps<E>, R: Reducer, TC: AttentionMatmulConfig>(
    vals: &mut RowWise<E>,
    base: &RowWise<E>,
    data: &F,
    #[comptime] config: TC,
) {
    vals.copy_from(base);
    R::reduce::<E, F, RowMax, TC>(vals, data, config)
}

#[cube]
/// Strategy for reducing across units participating in the same row
pub trait Reducer: CubeType {
    /// Reduction algorithm, applied inplace in vals
    fn reduce<E: Float, F: FragmentOps<E>, RO: ReduceOp<E>, TC: AttentionMatmulConfig>(
        vals: &mut RowWise<E>,
        data: &F,
        #[comptime] config: TC,
    );
}

#[cube]
/// A reduction operation
pub trait ReduceOp<E: Float> {
    /// Applies the reduction on the elements of the same row held by the unit
    fn reduce_local<F: FragmentOps<E>>(data: &F) -> RowWise<E>;

    /// Applies the reduction on the elements of the same row held by the unit,
    /// and to the accumulator, and store in the accumulator
    fn reduce_local_accumulate<F: FragmentOps<E>>(data: &F, acc: &mut RowWise<E>);

    /// The basic operation on two single values
    fn reduce_step_scalar(a: E, b: E) -> E;

    /// Accumulates elem into acc.
    /// If mask is activated, the element gets masked prior to being accumulated
    fn reduce_step_rowwise(acc: &mut RowWise<E>, elem: &RowWise<E>, mask: bool);
}
