use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::tile::{MaskTile, Reducer};
use crate::components::{
    AttentionPrecision,
    attention_types::*,
    tile::{RowWise, RunningState, row_max},
};
use crate::components::{
    fragment::{RowwiseFormat, RowwiseFormatExpand},
    tile::row_sum,
};

use crate::components::fragment::FragmentAttention;

#[cube]
pub fn tile_softmax<AP: AttentionPrecision, FA: FragmentAttention<AP>, R: Reducer>(
    rowwise_softmax: &mut <FA as FragmentAttention<AP>>::SoftmaxRow,
    mask: &MaskTile<AP, FA>,
    state: &mut RunningState<SM<AP>>,
    max_placeholder: &mut RowWise<SM<AP>>,
    sum_placeholder: &mut RowWise<SM<AP>>,
    #[comptime] dk: u32,
    #[comptime] config: FA::Config,
) -> RowWise<SM<AP>> {
    FA::SoftmaxRow::scale_and_mask::<MaskTile<AP, FA>>(
        rowwise_softmax,
        SM::<AP>::new(comptime!(1.0 / (dk as f32).sqrt())),
        mask,
    );

    row_max::<SM<AP>, <FA as FragmentAttention<AP>>::SoftmaxRow, R, FA::Config>(
        max_placeholder,
        state.m(),
        rowwise_softmax,
        config,
    );

    rowwise_softmax.exp_diff(max_placeholder);

    row_sum::<SM<AP>, <FA as FragmentAttention<AP>>::SoftmaxRow, R, FA::Config>(
        sum_placeholder,
        rowwise_softmax,
        config,
    );

    let exp_m_diff = state.m().exp_diff(max_placeholder);

    let new_l = exp_m_diff.mul(state.l()).add(sum_placeholder);

    state.update(max_placeholder, &new_l);

    exp_m_diff
}
