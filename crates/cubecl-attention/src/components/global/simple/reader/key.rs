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
pub struct DummyKeyReader<AP: AttentionPrecision, G: GlobalAttentionConfig> {
    global_iter: GlobalIterator<Line<KG<AP>>>,

    #[cube(comptime)]
    _phantom: PhantomData<G>,
}

#[cube]
impl<AP: AttentionPrecision, G: GlobalAttentionConfig> DummyKeyReader<AP, G> {
    pub fn new(key: View<Line<KG<AP>>, Coords2d>, step: u32) -> Self {
        let global_iter = GlobalIterator::new(key, step, ViewDirection::Row, false);

        DummyKeyReader::<AP, G> {
            global_iter,
            _phantom: PhantomData,
        }
    }
}

#[cube]
impl<AP: AttentionPrecision, G: GlobalAttentionConfig> AttentionReader<KS<AP>, G>
    for DummyKeyReader<AP, G>
{
    type Stage = StridedStage<KS<AP>, AttentionTilingLayout>;

    fn init_stage(&mut self, #[comptime] config: G) -> Self::Stage {
        StridedStage::new(StageIdent::Rhs, config.score_stage_memory_config())
    }

    fn read_global(&mut self, stage: &mut Self::Stage, #[comptime] config: G) {
        // TODO this reader is bad
        if UNIT_POS_Y == 0 {
            let memory_config = config.global_memory_config(AttentionIdent::Key);

            let mut slice = stage.as_slice_mut(1u32);

            let tile_rows_load = memory_config.elements_in_tile_row;
            let tile_cols_load = memory_config.elements_in_tile_col;
            let partition_rows_load = memory_config.elements_in_stage_row / tile_rows_load;
            let partition_cols_load = memory_config.elements_in_stage_col / tile_cols_load;

            let units_per_tile_row = comptime!(config.plane_dim() / tile_rows_load);
            let tile_cols_per_unit = comptime!(div_ceil(tile_cols_load, units_per_tile_row));

            let row_load_in_tile = UNIT_POS_X / units_per_tile_row;
            let col_load_in_tile_start = (UNIT_POS_X % units_per_tile_row) * tile_cols_per_unit;

            // Assumes row tiling order
            let num_elements_per_tile = tile_rows_load * tile_cols_load;
            let tile_row_stride_store = partition_rows_load * num_elements_per_tile;
            let tile_col_stride_store = num_elements_per_tile;

            let layout = TiledLayout::new(memory_config);
            let view = self.global_iter.view().view(layout);

            #[unroll]
            for tile_row_load in 0..partition_rows_load {
                #[unroll]
                for tile_col_load in 0..partition_cols_load {
                    if row_load_in_tile < tile_rows_load {
                        #[unroll]
                        for i in 0..tile_cols_per_unit {
                            let col_load = col_load_in_tile_start + i;

                            if col_load < tile_cols_load {
                                let tile_row_store = tile_col_load;
                                let tile_col_store = tile_row_load;
                                let tile_row_store_offset = tile_row_store * tile_row_stride_store;
                                let tile_col_store_offset = tile_col_store * tile_col_stride_store;
                                let store_offset = tile_row_store_offset + tile_col_store_offset;

                                let index_load = row_load_in_tile * tile_cols_load + col_load;
                                let index_store = col_load * tile_rows_load + row_load_in_tile;

                                slice[index_store + store_offset] =
                                    Line::cast_from(view.read_checked((
                                        (tile_row_load, tile_col_load).runtime(),
                                        index_load,
                                    )));
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
