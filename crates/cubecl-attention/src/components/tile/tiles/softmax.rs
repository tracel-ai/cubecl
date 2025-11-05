use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::AttentionPrecision;
use crate::components::attention_types::*;
use crate::components::fragment::AccScoreFormat;
use crate::components::fragment::FragmentAttention;
use crate::components::fragment::FragmentAttentionConfig;
use crate::components::fragment::{RowwiseFormat, RowwiseFormatExpand};
use crate::components::tile::MaskTile;
use crate::components::tile::Reducer;
use crate::components::tile::RowWise;
use crate::components::tile::RunningState;
use crate::components::tile::row_sum;

type RowwiseSoftmax<FA, AP> =
    <<FA as FragmentAttention<AP>>::SoftmaxScore as AccScoreFormat<SM<AP>>>::RowWiseFormat;

// #[derive(CubeType)]
// /// Softmax tile for the Tile Attention
// ///
// /// This tile is neither an input nor an output,
// /// but the intermediate step where the softmax part of attention happens
// pub struct SoftmaxTile<AP: AttentionPrecision, FA: FragmentAttention<AP>> {
//     pub flash: FA::SoftmaxScore,
// }

// #[cube]
// impl<AP: AttentionPrecision, FA: FragmentAttention<AP>> SoftmaxTile<AP, FA> {
//     pub fn new(#[comptime] config: FA::Config) -> Self {
//         let mut flash = FA::allocate_softmax(config);
//         FA::zero_softmax(&mut flash, config);

//         SoftmaxTile::<AP, FA> { flash }
//     }
// }

/// Scale the tile by a constant factor and apply the mask
#[cube]
pub fn scale_and_mask<AP: AttentionPrecision, FA: FragmentAttention<AP>>(
    rowwise_flash: &mut RowwiseSoftmax<FA, AP>,
    scale: SM<AP>,
    mask: &MaskTile<AP, FA>,
) {
    RowwiseSoftmax::<FA, AP>::scale_and_mask::<MaskTile<AP, FA>>(rowwise_flash, scale, mask);
}

// /// Compute the max of each row, starting with base
// /// as first element of the reduction, and storing result in placeholder
// pub fn row_max<E: Float, RF: RowwiseFormat<E>, R: Reducer, TC: FragmentAttentionConfig>(
//     rowwise_flash: &RF,
//     placeholder: &mut RowWise<E>,
//     base: &RowWise<E>,
//     #[comptime] config: TC,
// ) {
//     row_max::<E, RF, TC>(placeholder, base, rowwise_flash, config)
// }

/// Converts scores into (unnormalized) probabilities, updates running state,
/// and returns the factor needed to scale the accumulator
#[cube]
pub fn to_prob<E: Float, RF: RowwiseFormat<E>, R: Reducer, TC: FragmentAttentionConfig>(
    rowwise_flash: &mut RF,
    state: &mut RunningState<E>,
    new_m: &RowWise<E>,
    rowsum_placeholder: &mut RowWise<E>,
    #[comptime] config: TC,
) -> RowWise<E> {
    rowwise_flash.exp_diff(new_m);

    row_sum::<E, RF, R, TC>(rowsum_placeholder, &rowwise_flash, config);

    let exp_m_diff = state.m().exp_diff(new_m);

    let new_l = exp_m_diff.mul(state.l()).add(rowsum_placeholder);

    state.update(new_m, &new_l);

    exp_m_diff
}
