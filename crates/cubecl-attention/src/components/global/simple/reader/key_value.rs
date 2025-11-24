use crate::components::stage::AttentionTilingLayout;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::global::{
    GlobalReaderConfig, memory::GlobalIterator, read::tiled::TiledLayout,
};
use cubecl_matmul::components::stage::StridedStageMemory;
use cubecl_std::tensor::{View, layout::Coords2d};
use std::marker::PhantomData;

#[derive(CubeType)]
pub struct DummyKeyValueReader<EG: Float, ES: Float> {
    global_iter: GlobalIterator<Line<EG>>,
    #[cube(comptime)]
    config: GlobalReaderConfig,
    #[cube(comptime)]
    _phantom: PhantomData<ES>,
}

#[cube]
impl<EG: Float, ES: Float> DummyKeyValueReader<EG, ES> {
    pub fn new(
        view: View<Line<EG>, Coords2d>,
        step: u32,
        #[comptime] config: GlobalReaderConfig,
    ) -> Self {
        let global_iter = GlobalIterator::new(view, step, config.gmem_config.view_direction, false);

        DummyKeyValueReader::<EG, ES> {
            global_iter,
            config,
            _phantom: PhantomData,
        }
    }
}

#[cube]
impl<EG: Float, ES: Float> DummyKeyValueReader<EG, ES> {
    pub fn init_stage(&mut self) -> StridedStageMemory<ES, AttentionTilingLayout> {
        StridedStageMemory::new(self.config.smem_config)
    }

    pub fn read_global(&mut self, stage: &mut StridedStageMemory<ES, AttentionTilingLayout>) {
        if UNIT_POS_Y == 0 {
            // TODO this reader is bad, it's not coalesced
            let memory_config = self.config.smem_config;
            let mut slice = stage.as_slice_mut(1u32);

            let tile_rows = memory_config.elements_per_tile_along_row;
            let tile_cols = memory_config.elements_per_tile_along_col;
            let stage_rows = comptime!(memory_config.tiles_per_stage_along_row());
            let stage_cols = comptime!(memory_config.tiles_per_stage_along_col());

            let units_per_tile_row = comptime!(self.config.plane_dim / tile_rows);
            let tile_cols_per_unit = comptime!(div_ceil(tile_cols, units_per_tile_row));

            let row_in_tile = UNIT_POS_X / units_per_tile_row;
            let col_in_tile_start = (UNIT_POS_X % units_per_tile_row) * tile_cols_per_unit;

            // Assumes row tiling order
            let num_elements_per_tile = comptime!(tile_rows * tile_cols);
            let tile_row_stride = comptime!(stage_cols * num_elements_per_tile);
            let tile_col_stride = num_elements_per_tile;

            let layout = TiledLayout::new(self.config.smem_config);
            let view = self.global_iter.view().view(layout);

            #[unroll]
            for tile_row in 0..stage_rows {
                #[unroll]
                for tile_col in 0..stage_cols {
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

    pub fn advance_view(&mut self) {
        self.global_iter.advance();
    }
}
