use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::MatrixLayout;
use cubecl_matmul::components::tile::StridedTile;

use crate::components::AttentionPrecision;
use crate::components::attention_types::*;
use crate::components::tile::SparseArray;
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
    tmp_smem: SharedMemory<SM<AP>>,
    pub fragment: AM::Softmax,

    row: u32,
    col_start: u32,

    tmp_smem_start: u32,
    tmp_smem_end: u32,

    #[cube(comptime)]
    num_rows: u32,
    #[cube(comptime)]
    num_cols: u32,
    #[cube(comptime)]
    num_cols_per_unit: u32,
    #[cube(comptime)]
    config: AM::Config,
}

#[cube]
impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>> DummySoftmax<AP, AM> {
    pub fn new(#[comptime] config: AM::Config) -> Self {
        let mut fragment = AM::allocate_softmax(config);
        AM::zero_softmax(&mut fragment, config);

        let num_rows = config
            .attention_tile_size()
            .num_rows(AttentionIdent::Softmax);
        let num_cols = config
            .attention_tile_size()
            .num_cols(AttentionIdent::Softmax);
        let num_units_per_row = config.num_units_per_row(AttentionIdent::Softmax);
        let num_cols_per_unit = config.num_cols_per_unit(AttentionIdent::Softmax);

        let row = UNIT_POS_X / num_units_per_row;
        let col_start = (UNIT_POS_X % num_units_per_row) * num_cols_per_unit;

        let score_size = config.attention_tile_size().softmax_size();
        let tmp_smem_start = UNIT_POS_Y * score_size;
        let tmp_smem_end = tmp_smem_start + score_size;

        DummySoftmax::<AP, AM> {
            tmp_smem: SharedMemory::<SM<AP>>::new(score_size * config.num_planes()),
            fragment,
            row,
            col_start,
            tmp_smem_start,
            tmp_smem_end,
            num_rows,
            num_cols,
            num_cols_per_unit,
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
        let mut slice = self
            .tmp_smem
            .slice_mut(self.tmp_smem_start, self.tmp_smem_end)
            .try_cast_unchecked();

        AM::tmp_write_softmax(&self.fragment, &mut slice, self.config);

        if self.row < self.num_rows {
            #[unroll]
            for i in 0..self.num_cols_per_unit {
                let col = self.col_start + i;

                if col < self.num_cols {
                    let index = self.row * self.num_cols + col;

                    slice[index] =
                        slice[index] * Line::cast_from(scale) + mask.apply::<SM<AP>>(self.row, col);
                }
            }
        }

        sync_cube();

        let tile = StridedTile::<SM<AP>>::new_strided(
            slice.to_slice().try_cast_unchecked(),
            self.num_cols.runtime(),
            MatrixLayout::RowMajor,
        );
        AM::tmp_fill_prob(&tile, &mut self.fragment, self.config);
        sync_cube();
    }

    fn row_max(&self, placeholder: &mut Self::RowWise, base: &Self::RowWise) {
        Self::RowWise::row_max::<Self::PlaneLayout>(placeholder, base, &self.fragment)
    }

    fn to_prob(
        &mut self,
        state: &mut RunningState<Self::RowWise>,
        new_m: &Self::RowWise,
    ) -> Self::RowWise {
        let new_m_val = new_m.index(0u32);

        let mut slice = self
            .tmp_smem
            .slice_mut(self.tmp_smem_start, self.tmp_smem_end)
            .try_cast_unchecked();

        if self.row < self.num_rows {
            #[unroll]
            for i in 0..self.num_cols_per_unit {
                let col = self.col_start + i;

                if col < self.num_cols {
                    let index = self.row * self.num_cols + col;
                    slice[index] = Exp::exp(slice[index] - Line::cast_from(new_m_val));
                }
            }
        }

        sync_cube();

        let tile = StridedTile::<SM<AP>>::new_strided(
            slice.to_slice(),
            self.num_cols.runtime(),
            MatrixLayout::RowMajor,
        );
        AM::tmp_fill_prob(&tile, &mut self.fragment, self.config);

        sync_cube();
        let slice = self.tmp_smem.slice(self.tmp_smem_start, self.tmp_smem_end);

        let row_offset = self.row * self.num_cols;

        // let row_sum = row_sum::<SM<AP>, AM::Softmax>(&self.fragment);
        let mut row_sum = SM::<AP>::from_int(0);
        for i in 0..self.num_cols {
            row_sum += slice[row_offset + i];
        }

        let exp_m_diff = Exp::exp(state.m.index(0u32) - new_m_val);
        let new_l = exp_m_diff * state.l.index(0u32) + row_sum;

        state.update(new_m, &SparseArray::single(new_l));

        SparseArray::<SM<AP>>::single(exp_m_diff)
    }
}
