use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::AttentionPrecision;
use crate::components::attention_types::*;
use crate::components::tile::BroadcastReducer;
use crate::components::tile::RowWise;
use crate::components::tile::dummy::AttentionMatmulConfig;
use crate::components::tile::{PlaneLayout, PlaneLayoutExpand};
use crate::components::tile::{row_max, row_sum};
use crate::components::{
    TileMask,
    tile::{RunningState, SoftmaxTile, SoftmaxTileExpand, dummy::AttentionMatmul},
};

#[derive(CubeType)]
pub struct DummySoftmax<AP: AttentionPrecision, AM: AttentionMatmul<AP>> {
    pub fragment: AM::Softmax,

    #[cube(comptime)]
    config: AM::Config,
}

#[cube]
impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>> DummySoftmax<AP, AM> {
    pub fn new(#[comptime] config: AM::Config) -> Self {
        let mut fragment = AM::allocate_softmax(config);
        AM::zero_softmax(&mut fragment, config);

        DummySoftmax::<AP, AM> { fragment, config }
    }
}

#[cube]
impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>> SoftmaxTile<AP> for DummySoftmax<AP, AM> {
    type PlaneLayout = AM::Softmax;

    fn init_state(#[comptime] num_rows: u32) -> RunningState<SM<AP>> {
        RunningState::<SM<AP>>::init(num_rows)
    }

    fn init_placeholder(#[comptime] num_rows: u32) -> RowWise<SM<AP>> {
        RowWise::new_filled(num_rows, SM::<AP>::min_value())
    }

    fn zero(&mut self) {
        AM::zero_softmax(&mut self.fragment, self.config);
    }

    fn scale_and_mask(&mut self, scale: SM<AP>, mask: TileMask) {
        self.fragment.scale_and_mask(scale, mask);
    }

    fn row_max<TC: AttentionMatmulConfig>(
        &self,
        placeholder: &mut RowWise<SM<AP>>,
        base: &RowWise<SM<AP>>,
        #[comptime] config: TC,
    ) {
        row_max::<SM<AP>, Self::PlaneLayout, BroadcastReducer, TC>(
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
        self.fragment.exp_m_diff(new_m);

        row_sum::<SM<AP>, Self::PlaneLayout, BroadcastReducer, TC>(
            rowsum_placeholder,
            &self.fragment,
            config,
        );

        let exp_m_diff = state.m.exp_m_diff(new_m);

        let new_l = exp_m_diff.mul(&state.l).add(rowsum_placeholder);

        state.update(new_m, &new_l);

        exp_m_diff
    }
}
