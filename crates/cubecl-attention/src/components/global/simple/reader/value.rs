use crate::components::attention_types::*;
use crate::components::global::simple::reader::{AttentionReader, AttentionReaderExpand};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::StageIdent;
use cubecl_matmul::components::global::{
    memory::{GlobalIterator, ViewDirection},
    read::tiled::TiledLayout,
};
use cubecl_matmul::components::stage::StridedStage;
use cubecl_std::tensor::{View, layout::Coords2d};
use std::marker::PhantomData;

use crate::components::global::base::GlobalAttentionConfig;
use crate::components::tile::AttentionTilingLayout;
use crate::components::{AttentionIdent, AttentionPrecision};

#[derive(CubeType)]
pub struct DummyValueReader<AP: AttentionPrecision, G: GlobalAttentionConfig> {
    global_iter: GlobalIterator<Line<VG<AP>>>,

    #[cube(comptime)]
    _phantom: PhantomData<G>,
}

#[cube]
impl<AP: AttentionPrecision, G: GlobalAttentionConfig> DummyValueReader<AP, G> {
    pub fn new(value: View<Line<VG<AP>>, Coords2d>, step: u32) -> Self {
        let global_iter = GlobalIterator::new(value, step, ViewDirection::Row, false);

        DummyValueReader::<AP, G> {
            global_iter,
            _phantom: PhantomData,
        }
    }
}

#[cube]
impl<AP: AttentionPrecision, G: GlobalAttentionConfig> AttentionReader<VS<AP>, G>
    for DummyValueReader<AP, G>
{
    type Stage = StridedStage<VS<AP>, AttentionTilingLayout>;

    fn init_stage(&mut self, #[comptime] config: G) -> Self::Stage {
        StridedStage::new(StageIdent::Rhs, config.value_stage_memory_config())
    }

    fn read_global(&mut self, stage: &mut Self::Stage, #[comptime] config: G) {
        if UNIT_POS_Y == 0 {
            // TODO this reader is bad, it's not coalesced
            let memory_config = config.global_memory_config(AttentionIdent::Value);
            let mut slice = stage.as_slice_mut(1u32);

            let tile_rows = memory_config.elements_in_tile_row;
            let tile_cols = memory_config.elements_in_tile_col;
            let partition_rows = memory_config.elements_in_stage_row / tile_rows;
            let partition_cols = memory_config.elements_in_stage_col / tile_cols;

            let units_per_tile_row = comptime!(config.plane_dim() / tile_rows);
            let tile_cols_per_unit = comptime!(div_ceil(tile_cols, units_per_tile_row));

            let row_in_tile = UNIT_POS_X / units_per_tile_row;
            let col_in_tile_start = (UNIT_POS_X % units_per_tile_row) * tile_cols_per_unit;

            // Assumes row tiling order
            let num_elements_per_tile = tile_rows * tile_cols;
            let tile_row_stride = partition_cols * num_elements_per_tile;
            let tile_col_stride = num_elements_per_tile;

            let layout = TiledLayout::new(memory_config);
            let view = self.global_iter.view().view(layout);

            #[unroll]
            for tile_row in 0..partition_rows {
                #[unroll]
                for tile_col in 0..partition_cols {
                    if row_in_tile < tile_rows {
                        #[unroll]
                        for i in 0..tile_cols_per_unit {
                            let col = col_in_tile_start + i;

                            if col < tile_cols {
                                let tile_row_offset = tile_row * tile_row_stride;
                                let tile_col_offset = tile_col * tile_col_stride;
                                let offset = tile_row_offset + tile_col_offset;

                                let index = row_in_tile * tile_cols + col;

                                slice[index + offset] = Line::cast_from(
                                    view.read_checked(((tile_row, tile_col).runtime(), index)),
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    fn advance_view(&mut self) {
        self.global_iter.advance();
    }
}
