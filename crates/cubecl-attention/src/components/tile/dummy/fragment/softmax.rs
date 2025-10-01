use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::AttentionPrecision;
use crate::components::attention_types::*;
use crate::components::tile::SparseArray;
use crate::components::tile::{PlaneLayout, PlaneLayoutExpand};
use crate::components::tile::{RowWise, RowWiseExpand};
use crate::components::{
    AttentionIdent, TileMask,
    tile::{
        RunningState, SoftmaxTile, SoftmaxTileExpand,
        dummy::{AttentionMatmul, AttentionMatmulConfig},
    },
};

#[derive(CubeType)]
pub struct DummySoftmax<AP: AttentionPrecision, AM: AttentionMatmul<AP>> {
    pub fragment: AM::Softmax,

    row: u32,
    col_start: u32,

    #[cube(comptime)]
    num_cols: u32,
    #[cube(comptime)]
    config: AM::Config,
}

#[cube]
impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>> DummySoftmax<AP, AM> {
    pub fn new(#[comptime] config: AM::Config) -> Self {
        let mut fragment = AM::allocate_softmax(config);
        AM::zero_softmax(&mut fragment, config);

        let num_cols = config
            .attention_tile_size()
            .num_cols(AttentionIdent::Softmax);
        let num_units_per_row = config.num_units_per_row(AttentionIdent::Softmax);
        let num_cols_per_unit = config.num_cols_per_unit(AttentionIdent::Softmax);

        let row = UNIT_POS_X / num_units_per_row;
        let col_start = (UNIT_POS_X % num_units_per_row) * num_cols_per_unit;

        DummySoftmax::<AP, AM> {
            fragment,
            row,
            col_start,
            num_cols,
            config,
        }
    }
}

#[cube]
impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>> SoftmaxTile<AP> for DummySoftmax<AP, AM> {
    type PlaneLayout = AM::Softmax;
    type RowWise = SparseArray<SM<AP>>;

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
        for c in 0..self.num_cols {
            self.fragment.scale_at_coor(0u32, c, scale);
        }

        // let mut slice = self
        //     .tmp_smem
        //     .slice_mut(self.tmp_smem_start, self.tmp_smem_end)
        //     .try_cast_unchecked();

        // AM::tmp_write_softmax(&self.fragment, &mut slice, self.config);

        // if self.row < self.num_rows {
        //     #[unroll]
        //     for i in 0..self.num_cols_per_unit {
        //         let col = self.col_start + i;

        //         if col < self.num_cols {
        //             let index = self.row * self.num_cols + col;

        //             slice[index] =
        //                 slice[index] * Line::cast_from(scale) + mask.apply::<SM<AP>>(self.row, col);
        //         }
        //     }
        // }

        // sync_cube();

        // let tile = StridedTile::<SM<AP>>::new_strided(
        //     slice.to_slice().try_cast_unchecked(),
        //     self.num_cols.runtime(),
        //     MatrixLayout::RowMajor,
        // );
        // AM::tmp_fill_prob(&tile, &mut self.fragment, self.config);
        // sync_cube();
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
        let new_m_val = new_m.index(self.row);

        #[unroll]
        for c in 0..self.num_cols {
            self.fragment.exp_m_diff_at_coor(0u32, c, new_m_val);
        }

        Self::RowWise::row_sum::<Self::PlaneLayout>(rowsum_placeholder, &self.fragment);

        let exp_m_diff = Exp::exp(state.m.index(self.row) - new_m_val);
        let new_l = exp_m_diff * state.l.index(self.row) + rowsum_placeholder.index(self.row);

        state.update(new_m, &SparseArray::single(new_l));

        SparseArray::<SM<AP>>::single(exp_m_diff)

        // SparseArray::<SM<AP>>::new_zero(8u32)
    }
}
