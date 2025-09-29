use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::MatrixLayout;
use cubecl_matmul::components::tile::StridedTile;

use crate::components::AttentionIdent;
use crate::components::AttentionPrecision;
use crate::components::attention_types::*;
use crate::components::tile::AccumulatorTile;
use crate::components::tile::AccumulatorTileExpand;
use crate::components::tile::RowWise;
use crate::components::tile::ScaleMode;
use crate::components::tile::dummy::AttentionMatmul;
use crate::components::tile::dummy::AttentionMatmulConfig;

#[derive(CubeType)]
pub struct DummyAccumulator<AP: AttentionPrecision, AM: AttentionMatmul<AP>> {
    tmp_smem: SharedMemory<ACC<AP>>,
    pub fragment: AM::Accumulator,

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
impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>> DummyAccumulator<AP, AM> {
    pub fn new(#[comptime] config: AM::Config) -> DummyAccumulator<AP, AM> {
        let mut fragment = AM::allocate_accumulator(config);
        AM::zero_accumulator(&mut fragment, config);

        let num_rows = config.attention_tile_size().num_rows(AttentionIdent::Out);
        let num_cols = config.attention_tile_size().num_cols(AttentionIdent::Out);
        let num_units_per_row = config.num_units_per_row(AttentionIdent::Out);
        let num_cols_per_unit = config.num_cols_per_unit(AttentionIdent::Out);

        let row = UNIT_POS_X / num_units_per_row;
        let col_start = (UNIT_POS_X % num_units_per_row) * num_cols_per_unit;

        let acc_size = config.attention_tile_size().accumulator_size();
        let tmp_smem_start = UNIT_POS_Y * acc_size;
        let tmp_smem_end = tmp_smem_start + acc_size;

        DummyAccumulator::<AP, AM> {
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
impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>> AccumulatorTile<ACC<AP>>
    for DummyAccumulator<AP, AM>
{
    fn scale(&mut self, scale: &RowWise<ACC<AP>>, #[comptime] scale_op: ScaleMode) {
        let mut slice = self
            .tmp_smem
            .slice_mut(self.tmp_smem_start, self.tmp_smem_end)
            .try_cast_unchecked();

        AM::write_results::<ACC<AP>>(&self.fragment, &mut slice, self.config);

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

        let tile = StridedTile::<ACC<AP>>::new_strided(
            slice.to_slice(),
            self.num_cols.runtime(),
            MatrixLayout::RowMajor,
        );

        AM::tmp_fill_accumulator(&tile, &mut self.fragment, self.config);
    }
}
