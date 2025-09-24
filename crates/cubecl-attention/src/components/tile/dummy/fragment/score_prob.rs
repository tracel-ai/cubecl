use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::MatrixLayout;
use cubecl_matmul::components::tile::Tile;

use crate::components::{
    FlashIdent, TileMask,
    tile::dummy::{FlashMatmul, FlashMatmulConfig, FlashPrecision},
};

#[derive(CubeType)]
pub struct ScoreFragment<FP: FlashPrecision, FM: FlashMatmul<FP>> {
    tmp_smem: SharedMemory<FP::SP>,
    pub fragment: FM::ScoreProb,

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

        let score_size = config.attention_tile_size().score_prob_size();
        let tmp_smem_start = UNIT_POS_Y * score_size;
        let tmp_smem_end = tmp_smem_start + score_size;

        ScoreFragment::<FP, FM> {
            tmp_smem: SharedMemory::<FP::SP>::new(score_size * config.num_planes()),
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

    pub fn multiply_score(&mut self, factor: FP::SP) {
        let mut slice = self
            .tmp_smem
            .slice_mut(self.tmp_smem_start, self.tmp_smem_end)
            .try_cast_unchecked();

        FM::tmp_write_score_prob(&self.fragment, &mut slice, self.config);

        if self.row < self.num_rows {
            #[unroll]
            for i in 0..self.num_cols_per_unit {
                let col = self.col_start + i;

                if col < self.num_cols {
                    slice[self.row * self.num_cols + col] =
                        slice[self.row * self.num_cols + col] * Line::cast_from(factor);
                }
            }
        }

        sync_cube();
    }

    pub fn row_max(&mut self, base: FP::SP) -> FP::SP {
        let slice = self
            .tmp_smem
            .slice_mut(self.tmp_smem_start, self.tmp_smem_end);

        let row_offset = self.row * self.num_cols;
        let mut rowmax = base;

        for i in 0..self.num_cols {
            let ts = slice[row_offset + i];
            if ts > rowmax {
                rowmax = ts;
            }
        }

        rowmax
    }

    pub fn to_prob(&mut self, m: FP::SP) {
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
                    slice[index] = Exp::exp(slice[index] - Line::cast_from(m));
                }
            }
        }

        sync_cube();

        let tile = Tile::<FP::SP> {
            slice: slice.to_slice(),
            stride: self.num_cols.runtime(),
            layout: MatrixLayout::RowMajor,
        };
        FM::tmp_fill_prob(&tile, &mut self.fragment, self.config);
    }

    pub fn row_sum(&self) -> FP::SP {
        let slice = self.tmp_smem.slice(self.tmp_smem_start, self.tmp_smem_end);

        let row_offset = self.row * self.num_cols;

        let mut rowsum = FP::SP::from_int(0);
        for i in 0..self.num_cols {
            rowsum += slice[row_offset + i];
        }

        rowsum
    }

    pub fn apply_mask(&mut self, mask: TileMask) {
        let mut slice: SliceMut<Line<FP::SP>> = self
            .tmp_smem
            .slice_mut(self.tmp_smem_start, self.tmp_smem_end)
            .try_cast_unchecked();

        sync_cube();
        if self.row < self.num_rows {
            #[unroll]
            for i in 0..self.num_cols_per_unit {
                let col = self.col_start + i;

                if col < self.num_cols {
                    let index = self.row * self.num_cols + col;
                    slice[index] = slice[index] + mask.apply::<FP::SP>(self.row, col);
                }
            }
        }

        sync_cube();

        let tile = Tile::<FP::SP> {
            slice: slice.to_slice().try_cast_unchecked(),
            stride: self.num_cols.runtime(),
            layout: MatrixLayout::RowMajor,
        };
        FM::tmp_fill_prob(&tile, &mut self.fragment, self.config);
        sync_cube();
    }
}
