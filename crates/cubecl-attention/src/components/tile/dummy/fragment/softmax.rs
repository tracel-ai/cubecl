use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::AttentionPrecision;
use crate::components::attention_types::*;
use crate::components::tile::RowVals;
use crate::components::tile::{PlaneLayout, PlaneLayoutExpand};
use crate::components::tile::{RowWise, RowWiseExpand};
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
    type RowWise = RowVals<SM<AP>>;

    fn init_state(#[comptime] num_rows: u32) -> RunningState<Self::RowWise> {
        RunningState::<Self::RowWise>::init(num_rows)
    }

    fn init_placeholder(#[comptime] num_rows: u32) -> Self::RowWise {
        Self::RowWise::new_filled(num_rows, SM::<AP>::min_value())
    }

    fn zero(&mut self) {
        AM::zero_softmax(&mut self.fragment, self.config);
    }

    fn scale_and_mask(&mut self, scale: SM<AP>, mask: TileMask) {
        #[unroll]
        for c in 0..self.fragment.num_local_cols() {
            // TODO more than one row
            // TODO mask
            self.fragment.scale_at_coor(0u32, c, scale);
        }
    }

    fn row_max(&self, placeholder: &mut Self::RowWise, base: &Self::RowWise) {
        Self::RowWise::row_max::<Self::PlaneLayout>(placeholder, base, &self.fragment)
    }

    fn to_prob(
        &mut self,
        state: &mut RunningState<Self::RowWise>,
        new_m: &Self::RowWise,
        rowsum_placeholder: &mut Self::RowWise,
    ) -> Self::RowWise {
        let new_m_val = new_m.index(0u32);

        #[unroll]
        for c in 0..self.fragment.num_local_cols() {
            // TODO more than one row
            self.fragment.exp_m_diff_at_coor(0u32, c, new_m_val);
        }

        Self::RowWise::row_sum::<Self::PlaneLayout>(rowsum_placeholder, &self.fragment);

        let exp_m_diff = Exp::exp(state.m.index(0u32) - new_m_val);
        let new_l = exp_m_diff * state.l.index(0u32) + rowsum_placeholder.index(0u32);

        state.update(new_m, &RowVals::new_filled(1u32, new_l));

        RowVals::<SM<AP>>::new_filled(1u32, exp_m_diff)
    }
}
