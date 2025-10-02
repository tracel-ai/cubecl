use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::MatrixLayout;
use cubecl_matmul::components::tile::StridedTile;
use std::marker::PhantomData;

use crate::components::AttentionIdent;
use crate::components::AttentionPrecision;
use crate::components::attention_types::*;
use crate::components::tile::AccumulatorTile;
use crate::components::tile::AccumulatorTileExpand;
use crate::components::tile::ScaleMode;
use crate::components::tile::dummy::AttentionMatmul;
use crate::components::tile::dummy::AttentionMatmulConfig;
use crate::components::tile::row::{PlaneLayout, PlaneLayoutExpand};
use crate::components::tile::{RowWise, RowWiseExpand};

#[derive(CubeType)]
pub struct DummyAccumulator<AP: AttentionPrecision, AM: AttentionMatmul<AP>, RW: RowWise> {
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
    #[cube(comptime)]
    _phantom: PhantomData<RW>,
}

#[cube]
impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>, RW: RowWise> DummyAccumulator<AP, AM, RW> {
    pub fn new(#[comptime] config: AM::Config) -> DummyAccumulator<AP, AM, RW> {
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

        DummyAccumulator::<AP, AM, RW> {
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
            _phantom: PhantomData,
        }
    }
}

#[cube]
impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>, RW: RowWise> AccumulatorTile<AP, RW>
    for DummyAccumulator<AP, AM, RW>
{
    fn scale(&mut self, scale: &RW, #[comptime] scale_mode: ScaleMode) {
        let scale = ACC::<AP>::cast_from(scale.index(0u32));
        let scale = match scale_mode {
            ScaleMode::Multiply => scale,
            ScaleMode::Divide => Recip::recip(scale),
        };

        #[unroll]
        for c in 0..self.num_cols {
            self.fragment.scale_at_coor(0u32, c, scale);
        }
    }
}
