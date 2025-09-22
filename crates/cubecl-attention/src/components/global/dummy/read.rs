use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::global::{
    memory::{GlobalIterator, ViewDirection},
    read::tiled::TiledLayout,
};
use cubecl_matmul::components::stage::{FullStageReader, StageMemory};
use cubecl_matmul::components::tile::Tile;
use cubecl_matmul::components::{MatrixLayout, StageIdent};
use cubecl_std::tensor::{View, layout::Coords2d};
use std::marker::PhantomData;

use crate::components::global::base::GlobalAttentionConfig;
use crate::components::stage::StageAttentionConfig;
use crate::components::stage::dummy::AttentionStageMemoryConfig;
use crate::components::tile::AttentionTilingLayout;
use crate::components::{AttentionPrecision, FlashIdent};

#[derive(CubeType)]
pub struct QueryReader<AP: AttentionPrecision> {
    query: View<Line<AP::EI>, Coords2d>,
}

#[derive(CubeType)]
pub struct DummyKeyReader<AP: AttentionPrecision, G: GlobalAttentionConfig> {
    global_iter: GlobalIterator<AP::EI>,
    stage_memory: StageMemory<AP::ES, AttentionTilingLayout>,

    #[cube(comptime)]
    _phantom: PhantomData<G>,
}

#[derive(CubeType)]
pub struct DummyValueReader<AP: AttentionPrecision, G: GlobalAttentionConfig> {
    global_iter: GlobalIterator<AP::EI>,
    stage_memory: StageMemory<AP::ES, AttentionTilingLayout>,

    #[cube(comptime)]
    _phantom: PhantomData<G>,
}

#[cube]
impl<AP: AttentionPrecision> QueryReader<AP> {
    pub fn new(q_offset: u32, query: View<Line<AP::EI>, Coords2d>) -> Self {
        let query = query.slice((q_offset, 0), query.shape());

        QueryReader::<AP> { query }
    }

    pub fn get_tile<S: StageAttentionConfig>(
        &self,
        tile: Coords2d,
        #[comptime] config: S,
    ) -> Tile<AP::EI> {
        let (row_in_partition, col) = tile;
        let attention_tile_size = config.tiling_scheme().tile_size;

        let row = row_in_partition + UNIT_POS_Y * config.tiling_scheme().partition_size.seq_q;

        Tile::<AP::EI> {
            slice: self
                .query
                .slice(
                    (
                        row * attention_tile_size.seq_q,
                        col * attention_tile_size.head_dim,
                    ),
                    (attention_tile_size.seq_q, attention_tile_size.head_dim).runtime(),
                )
                .to_linear_slice(),
            stride: config.tiling_scheme().head_dim(),
            layout: MatrixLayout::RowMajor,
        }
    }
}

#[cube]
impl<AP: AttentionPrecision, G: GlobalAttentionConfig> DummyKeyReader<AP, G> {
    pub fn new(key: View<Line<AP::EI>, Coords2d>, step: u32, #[comptime] config: G) -> Self {
        let global_iter = GlobalIterator::new(key, step, ViewDirection::Row, false);
        let stage_memory = StageMemory::new::<AttentionStageMemoryConfig>(
            1u32,
            StageIdent::Rhs,
            config.score_stage_memory_config(),
        );

        DummyKeyReader::<AP, G> {
            global_iter,
            stage_memory,
            _phantom: PhantomData,
        }
    }

    pub fn stage_reader(&self) -> FullStageReader<AP::ES, AttentionTilingLayout> {
        FullStageReader::<AP::ES, AttentionTilingLayout> {
            stage_memory: self.stage_memory,
            stage_ident: StageIdent::Rhs,
        }
    }

    pub fn read_transposed(&mut self, #[comptime] config: G) {
        // TODO this reader is bad
        if UNIT_POS_Y == 0 {
            let memory_config = config.global_memory_config(FlashIdent::Key);

            let mut slice = self.stage_memory.as_slice_mut(1u32);

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

                                slice[index_store + store_offset] = Line::cast_from(
                                    view.read_checked(((tile_row_load, tile_col_load), index_load)),
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

#[cube]
impl<AP: AttentionPrecision, G: GlobalAttentionConfig> DummyValueReader<AP, G> {
    pub fn new(value: View<Line<AP::EI>, Coords2d>, step: u32, #[comptime] config: G) -> Self {
        let global_iter = GlobalIterator::new(value, step, ViewDirection::Row, false);
        let stage_memory = StageMemory::new::<AttentionStageMemoryConfig>(
            1u32,
            StageIdent::Rhs,
            config.value_stage_memory_config(),
        );

        DummyValueReader::<AP, G> {
            global_iter,
            stage_memory,
            _phantom: PhantomData,
        }
    }

    pub fn stage_reader(&self) -> FullStageReader<AP::ES, AttentionTilingLayout> {
        FullStageReader::<AP::ES, AttentionTilingLayout> {
            stage_memory: self.stage_memory,
            stage_ident: StageIdent::Rhs,
        }
    }

    pub fn read(&mut self, #[comptime] config: G) {
        if UNIT_POS_Y == 0 {
            // TODO this reader is bad, it's not coalesced
            let memory_config = config.global_memory_config(FlashIdent::Value);
            let mut slice = self.stage_memory.as_slice_mut(1u32);

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
                                    view.read_checked(((tile_row, tile_col), index)),
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
