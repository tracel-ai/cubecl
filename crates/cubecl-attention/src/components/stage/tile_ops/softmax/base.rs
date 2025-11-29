use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::stage::RowMax;
use crate::components::stage::RowSum;
use crate::components::tile::RowWise;
use crate::components::tile::RowwiseFormat;
use crate::components::tile::TileAttentionConfig;

use crate::components::stage::{MaskTile, RunningState};
use crate::components::tile::RowwiseFormatExpand;
use crate::components::{AttentionPrecision, attention_types::*};

use crate::components::tile::TileAttention;

#[cube]
/// Applies softmax to a tile with masking and updates the running state.
///
/// Scales by `1 / sqrt(head_dim)`, applies the mask, computes row-wise max and sum,
/// exponentiates, and updates the softmax state.
///
/// Returns the exponential difference used for normalization.
pub fn tile_softmax<AP: AttentionPrecision, TA: TileAttention<AP>, R: Reducer>(
    rowwise_softmax: &mut <TA as TileAttention<AP>>::SoftmaxRow,
    mask: &MaskTile<AP, TA>,
    state: &mut RunningState<SM<AP>>,
    max_placeholder: &mut RowWise<SM<AP>>,
    sum_placeholder: &mut RowWise<SM<AP>>,
    #[comptime] head_dim: u32,
    #[comptime] config: TA::Config,
) -> RowWise<SM<AP>> {
    TA::SoftmaxRow::scale_and_mask::<MaskTile<AP, TA>>(
        rowwise_softmax,
        SM::<AP>::new(comptime!(1.0 / (head_dim as f32).sqrt())),
        mask,
    );

    row_max::<SM<AP>, <TA as TileAttention<AP>>::SoftmaxRow, R, TA::Config>(
        max_placeholder,
        state.m(),
        rowwise_softmax,
        config,
    );

    rowwise_softmax.exp_diff(max_placeholder);

    row_sum::<SM<AP>, <TA as TileAttention<AP>>::SoftmaxRow, R, TA::Config>(
        sum_placeholder,
        rowwise_softmax,
        config,
    );

    let exp_m_diff = state.m().exp_diff(max_placeholder);

    let new_l = exp_m_diff.mul(state.l()).add(sum_placeholder);

    state.update(max_placeholder, &new_l);

    exp_m_diff
}

#[cube]
/// Computes the sum of rows on a fragment, using the Reducer's strategy
pub fn row_sum<E: Float, F: RowwiseFormat<E>, R: Reducer, TC: TileAttentionConfig>(
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
pub fn row_max<E: Float, F: RowwiseFormat<E>, R: Reducer, TC: TileAttentionConfig>(
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
    fn reduce<E: Float, F: RowwiseFormat<E>, RO: ReduceOp<E>, TC: TileAttentionConfig>(
        vals: &mut RowWise<E>,
        data: &F,
        #[comptime] config: TC,
    );
}

#[cube]
/// A reduction operation
pub trait ReduceOp<E: Float> {
    /// Applies the reduction on the elements of the same row held by the unit
    fn reduce_local<F: RowwiseFormat<E>>(data: &F) -> RowWise<E>;

    /// Applies the reduction on the elements of the same row held by the unit,
    /// and to the accumulator, and store in the accumulator
    fn reduce_local_accumulate<F: RowwiseFormat<E>>(data: &F, acc: &mut RowWise<E>);

    /// The basic operation on two single values
    fn reduce_step_scalar(a: E, b: E) -> E;

    /// Accumulates elem into acc.
    /// If mask is activated, the element gets masked prior to being accumulated
    fn reduce_step_rowwise(acc: &mut RowWise<E>, elem: &RowWise<E>, mask: bool);
}
