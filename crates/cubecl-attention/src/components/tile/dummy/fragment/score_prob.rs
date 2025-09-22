use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::MatrixLayout;
use cubecl_matmul::components::tile::Tile;

use crate::components::{
    FlashIdent,
    tile::dummy::{FlashMatmul, FlashMatmulConfig, FlashPrecision},
};

#[derive(CubeType)]
pub struct ScoreFragment<FP: FlashPrecision, FM: FlashMatmul<FP>> {
    tmp_smem: SharedMemory<FP::SP>,
    pub fragment: FM::ScoreProb,

    row: u32,
    col_start: u32,

    #[cube(comptime)]
    num_rows: u32,
    #[cube(comptime)]
    num_cols: u32,
    #[cube(comptime)]
    num_cols_per_unit: u32,
    #[cube(comptime)]
    config: FM::Config,
}

#[cube]
impl<FP: FlashPrecision, FM: FlashMatmul<FP>> ScoreFragment<FP, FM> {
    pub fn new(#[comptime] config: FM::Config) -> Self {
        let mut fragment = FM::allocate_score_prob(config);
        FM::zero_score_prob(&mut fragment, config);

        let num_rows = config.attention_tile_size().num_rows(FlashIdent::ScoreProb);
        let num_cols = config.attention_tile_size().num_cols(FlashIdent::ScoreProb);
        let num_units_per_row = config.num_units_per_row(FlashIdent::ScoreProb);
        let num_cols_per_unit = config.num_cols_per_unit(FlashIdent::ScoreProb);

        let row = UNIT_POS_X / num_units_per_row;
        let col_start = (UNIT_POS_X % num_units_per_row) * num_cols_per_unit;

        ScoreFragment::<FP, FM> {
            tmp_smem: SharedMemory::<FP::SP>::new(config.attention_tile_size().score_prob_size()),
            fragment,
            row,
            col_start,
            num_rows,
            num_cols,
            num_cols_per_unit,
            config,
        }
    }

    pub fn multiply_score(&mut self, factor: FP::SP) {
        FM::tmp_write_score_prob(
            &self.fragment,
            &mut self.tmp_smem.to_slice_mut().try_cast_unchecked(),
            self.config,
        );

        if self.row < self.num_rows {
            #[unroll]
            for i in 0..self.num_cols_per_unit {
                let col = self.col_start + i;

                if col < self.num_cols {
                    self.tmp_smem[self.row * self.num_cols + col] *= factor;
                }
            }
        }

        sync_cube();
    }

    pub fn row_max(&mut self, base: FP::SP) -> FP::SP {
        let row_offset = self.row * self.num_cols;
        let mut rowmax = base;

        for i in 0..self.num_cols {
            let ts = self.tmp_smem[row_offset + i];
            if ts > rowmax {
                rowmax = ts;
            }
        }

        rowmax
    }

    pub fn to_prob(&mut self, m: FP::SP) {
        if self.row < self.num_rows {
            #[unroll]
            for i in 0..self.num_cols_per_unit {
                let col = self.col_start + i;

                if col < self.num_cols {
                    let index = self.row * self.num_cols + col;
                    self.tmp_smem[index] = Exp::exp(self.tmp_smem[index] - m);
                }
            }
        }

        sync_cube();

        let tile = Tile::<FP::SP> {
            slice: self.tmp_smem.to_slice().try_cast_unchecked(),
            stride: self.num_cols.runtime(),
            layout: MatrixLayout::RowMajor,
        };
        FM::tmp_fill_prob(&tile, &mut self.fragment, self.config);
    }

    pub fn row_sum(&self) -> FP::SP {
        let row_offset = self.row * self.num_cols;

        let mut rowsum = FP::SP::from_int(0);
        for i in 0..self.num_cols {
            rowsum += self.tmp_smem[row_offset + i];
        }

        rowsum
    }

    pub fn apply_mask(&mut self, row_col_remove: (u32, u32)) {
        sync_cube();
        if self.row < self.num_rows {
            #[unroll]
            for i in 0..self.num_cols_per_unit {
                let col = self.col_start + i;

                if col < self.num_cols && (self.row >= row_col_remove.0 || col >= row_col_remove.1)
                {
                    let index = self.row * self.num_cols + col;
                    self.tmp_smem[index] = FP::SP::from_int(-9999999999);
                }
            }
        }

        sync_cube();

        let tile = Tile::<FP::SP> {
            slice: self.tmp_smem.to_slice().try_cast_unchecked(),
            stride: self.num_cols.runtime(),
            layout: MatrixLayout::RowMajor,
        };
        FM::tmp_fill_prob(&tile, &mut self.fragment, self.config);
        sync_cube();
    }
}
