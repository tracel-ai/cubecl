use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::MatrixLayout;
use cubecl_matmul::components::tile::StridedTile;

use crate::components::FlashIdent;
use crate::components::tile::AccumulatorTile;
use crate::components::tile::AccumulatorTileExpand;
use crate::components::tile::RowWise;
use crate::components::tile::ScaleMode;
use crate::components::tile::dummy::FlashMatmul;
use crate::components::tile::dummy::FlashMatmulConfig;
use crate::components::tile::dummy::FlashPrecision;

#[derive(CubeType)]
pub struct DummyAccumulator<FP: FlashPrecision, FM: FlashMatmul<FP>> {
    tmp_smem: SharedMemory<FP::A>,
    pub fragment: FM::Accumulator,

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
impl<FP: FlashPrecision, FM: FlashMatmul<FP>> DummyAccumulator<FP, FM> {
    pub fn new(#[comptime] config: FM::Config) -> DummyAccumulator<FP, FM> {
        let mut fragment = FM::allocate_accumulator(config);
        FM::zero_accumulator(&mut fragment, config);

        let num_rows = config.attention_tile_size().num_rows(FlashIdent::Out);
        let num_cols = config.attention_tile_size().num_cols(FlashIdent::Out);
        let num_units_per_row = config.num_units_per_row(FlashIdent::Out);
        let num_cols_per_unit = config.num_cols_per_unit(FlashIdent::Out);

        let row = UNIT_POS_X / num_units_per_row;
        let col_start = (UNIT_POS_X % num_units_per_row) * num_cols_per_unit;

        let acc_size = config.attention_tile_size().accumulator_size();
        let tmp_smem_start = UNIT_POS_Y * acc_size;
        let tmp_smem_end = tmp_smem_start + acc_size;

        DummyAccumulator::<FP, FM> {
            tmp_smem: SharedMemory::new(acc_size * config.num_planes()),
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
impl<FP: FlashPrecision, FM: FlashMatmul<FP>> AccumulatorTile<FP::A> for DummyAccumulator<FP, FM> {
    fn scale(&mut self, scale: &RowWise<FP::A>, #[comptime] scale_op: ScaleMode) {
        let mut slice = self
            .tmp_smem
            .slice_mut(self.tmp_smem_start, self.tmp_smem_end)
            .try_cast_unchecked();

        FM::write_results::<FP::A>(&self.fragment, &mut slice, self.config);

        if self.row < self.num_rows {
            #[unroll]
            for i in 0..self.num_cols_per_unit {
                let col = self.col_start + i;

                if col < self.num_cols {
                    match scale_op {
                        ScaleMode::Multiply => {
                            slice[self.row * self.num_cols + col] = slice
                                [self.row * self.num_cols + col]
                                * Line::cast_from(scale.index(0u32))
                        }
                        ScaleMode::Divide => {
                            slice[self.row * self.num_cols + col] = slice
                                [self.row * self.num_cols + col]
                                / Line::cast_from(scale.index(0u32))
                        }
                    }
                }
            }
        }

        let tile = StridedTile::<FP::A> {
            slice: slice.to_slice(),
            stride: self.num_cols.runtime(),
            layout: MatrixLayout::RowMajor,
        };

        FM::tmp_fill_accumulator(&tile, &mut self.fragment, self.config);
    }
}
