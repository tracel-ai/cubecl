use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::MatrixLayout;
use cubecl_matmul::components::tile::Tile;

use crate::components::FlashIdent;
use crate::components::tile::dummy::FlashMatmul;
use crate::components::tile::dummy::FlashMatmulConfig;
use crate::components::tile::dummy::FlashPrecision;

#[derive(CubeType)]
pub struct AccumulatorFragment<FP: FlashPrecision, FM: FlashMatmul<FP>> {
    tmp_smem: SharedMemory<FP::A>,
    pub fragment: FM::Accumulator,

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
impl<FP: FlashPrecision, FM: FlashMatmul<FP>> AccumulatorFragment<FP, FM> {
    pub fn new(#[comptime] config: FM::Config) -> AccumulatorFragment<FP, FM> {
        let mut fragment = FM::allocate_accumulator(config);
        FM::zero_accumulator(&mut fragment, config);

        let num_rows = config.attention_tile_size().num_rows(FlashIdent::Out);
        let num_cols = config.attention_tile_size().num_cols(FlashIdent::Out);
        let num_units_per_row = config.num_units_per_row(FlashIdent::Out);
        let num_cols_per_unit = config.num_cols_per_unit(FlashIdent::Out);

        let row = UNIT_POS_X / num_units_per_row;
        let col_start = (UNIT_POS_X % num_units_per_row) * num_cols_per_unit;

        AccumulatorFragment::<FP, FM> {
            tmp_smem: SharedMemory::new(config.attention_tile_size().accumulator_size()),
            fragment,
            row,
            col_start,
            num_rows,
            num_cols,
            num_cols_per_unit,
            config,
        }
    }

    pub fn scale(&mut self, factor: FP::A) {
        FM::write_results::<FP::A>(
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

        let tile = Tile::<FP::A> {
            slice: self.tmp_smem.to_slice().try_cast_unchecked(),
            stride: self.num_cols.runtime(),
            layout: MatrixLayout::RowMajor,
        };

        FM::tmp_fill_accumulator(&tile, &mut self.fragment, self.config);
    }
}
