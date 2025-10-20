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
use crate::components::tile::{RunningState, SoftmaxTile, SoftmaxTileExpand};
use crate::components::tile::{row_max, row_sum};

#[derive(CubeType)]
pub struct DummySoftmax<AP: AttentionPrecision, AM: AttentionMatmul<AP>> {
    pub fragment: AM::Softmax,
}

#[cube]
impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>> DummySoftmax<AP, AM> {
    pub fn new(#[comptime] config: AM::Config) -> Self {
        let mut fragment = AM::allocate_softmax(config);
        AM::zero_softmax(&mut fragment, config);

        DummySoftmax::<AP, AM> { fragment }
    }
}

#[cube]
impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>> SoftmaxTile<AP> for DummySoftmax<AP, AM> {
    type Fragment = AM::Softmax;

    fn init_state(#[comptime] num_rows: u32) -> RunningState<SM<AP>> {
        RunningState::<SM<AP>>::init(num_rows)
    }

    fn scale_and_mask<M: MaskTile>(this: &mut Self, scale: SM<AP>, mask: &M) {
        Self::Fragment::scale_and_mask::<M>(&mut this.fragment, scale, mask);
    }

    fn row_max<TC: AttentionMatmulConfig>(
        &self,
        placeholder: &mut RowWise<SM<AP>>,
        base: &RowWise<SM<AP>>,
        #[comptime] config: TC,
    ) {
        row_max::<SM<AP>, Self::Fragment, BroadcastReducer, TC>(
            placeholder,
            base,
            &self.fragment,
            config,
        )
    }

    fn to_prob<TC: AttentionMatmulConfig>(
        &mut self,
        state: &mut RunningState<SM<AP>>,
        new_m: &RowWise<SM<AP>>,
        rowsum_placeholder: &mut RowWise<SM<AP>>,
        #[comptime] config: TC,
    ) -> RowWise<SM<AP>> {
        self.fragment.exp_diff(new_m);

        row_sum::<SM<AP>, Self::Fragment, BroadcastReducer, TC>(
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
