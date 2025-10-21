use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::AttentionPrecision;
use crate::components::attention_types::*;
use crate::components::fragment::AttentionMatmul;
use crate::components::fragment::AttentionMatmulConfig;
use crate::components::fragment::{FragmentOps, FragmentOpsExpand};
use crate::components::tile::BroadcastReducer;
use crate::components::tile::MaskTile;
use crate::components::tile::RowWise;
use crate::components::tile::RunningState;
use crate::components::tile::{row_max, row_sum};

#[derive(CubeType)]
/// Softmax tile for the Tile Attention
///
/// This tile is neither an input nor an output,
/// but the intermediate step where the softmax part of attention happens
pub struct SoftmaxTile<AP: AttentionPrecision, AM: AttentionMatmul<AP>> {
    pub fragment: AM::Softmax,
}

#[cube]
impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>> SoftmaxTile<AP, AM> {
    pub fn new(#[comptime] config: AM::Config) -> Self {
        let mut fragment = AM::allocate_softmax(config);
        AM::zero_softmax(&mut fragment, config);

        SoftmaxTile::<AP, AM> { fragment }
    }

    /// Init the running state used in softmax
    pub fn init_state(#[comptime] num_rows: u32) -> RunningState<SM<AP>> {
        RunningState::<SM<AP>>::init(num_rows)
    }

    /// Scale the tile by a constant factor and apply the mask
    pub fn scale_and_mask(&mut self, scale: SM<AP>, mask: &MaskTile<AP, AM>) {
        AM::Softmax::scale_and_mask::<MaskTile<AP, AM>>(&mut self.fragment, scale, mask);
    }

    /// Compute the max of each row, starting with base
    /// as first element of the reduction, and storing result in placeholder
    pub fn row_max<TC: AttentionMatmulConfig>(
        &self,
        placeholder: &mut RowWise<SM<AP>>,
        base: &RowWise<SM<AP>>,
        #[comptime] config: TC,
    ) {
        row_max::<SM<AP>, AM::Softmax, BroadcastReducer, TC>(
            placeholder,
            base,
            &self.fragment,
            config,
        )
    }

    /// Converts scores into (unnormalized) probabilities, updates running state,
    /// and returns the factor needed to scale the accumulator
    pub fn to_prob<TC: AttentionMatmulConfig>(
        &mut self,
        state: &mut RunningState<SM<AP>>,
        new_m: &RowWise<SM<AP>>,
        rowsum_placeholder: &mut RowWise<SM<AP>>,
        #[comptime] config: TC,
    ) -> RowWise<SM<AP>> {
        self.fragment.exp_diff(new_m);

        row_sum::<SM<AP>, AM::Softmax, BroadcastReducer, TC>(
            rowsum_placeholder,
            &self.fragment,
            config,
        );

        let exp_m_diff = state.m().exp_diff(new_m);

        let new_l = exp_m_diff.mul(state.l()).add(rowsum_placeholder);

        state.update(new_m, &new_l);

        exp_m_diff
    }
}
