use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::global::memory::{TensorReader, ViewDirection};
use cubecl_matmul::components::stage::{FullStageToTileReader, StageMemory};
use cubecl_matmul::components::tile::Tile;
use cubecl_matmul::components::{MatrixLayout, StageIdent};
use cubecl_std::tensor::{View, layout::Coords3d};
use std::marker::PhantomData;

use crate::components::global::base::GlobalAttentionConfig;
use crate::components::stage::StageAttentionConfig;
use crate::components::stage::dummy::AttentionStageMemoryConfig;
use crate::components::tile::AttentionTilingLayout;
use crate::components::tile::dummy::FlashMatmulConfig;
use crate::components::{AttentionPrecision, FlashIdent};

#[derive(CubeType)]
pub struct QueryLoader<AP: AttentionPrecision> {
    tensor_reader: TensorReader<AP::EI>,
}

#[derive(CubeType)]
pub struct DummyKeyLoader<AP: AttentionPrecision, G: GlobalAttentionConfig> {
    tensor_reader: TensorReader<AP::EI>,
    stage_memory: StageMemory<AP::ES, AttentionTilingLayout>,

    #[cube(comptime)]
    _phantom: PhantomData<G>,
}

#[derive(CubeType)]
pub struct DummyValueLoader<AP: AttentionPrecision, G: GlobalAttentionConfig> {
    tensor_reader: TensorReader<AP::EI>,
    stage_memory: StageMemory<AP::ES, AttentionTilingLayout>,

    #[cube(comptime)]
    _phantom: PhantomData<G>,
}

#[cube]
impl<AP: AttentionPrecision> QueryLoader<AP> {
    pub fn new(q_offset: u32, query: View<Line<AP::EI>, Coords3d>) -> Self {
        let tensor_reader = TensorReader::new(query, (0u32.runtime(), q_offset, 0u32.runtime()));

        QueryLoader::<AP> { tensor_reader }
    }

    pub fn get_tile<S: StageAttentionConfig>(
        &self,
        row: u32,
        col: u32,
        #[comptime] config: S,
    ) -> Tile<AP::EI> {
        let attention_tile_size = config.tiling_scheme().tile_size;
        let tile = Tile::<AP::EI> {
            slice: self.tensor_reader.view.slice(
                (
                    // batch
                    0u32.runtime(),
                    self.tensor_reader.row_offset.read() + row * attention_tile_size.seq_q,
                    col * attention_tile_size.head_dim,
                ),
                attention_tile_size.query_size(),
            ),
            stride: attention_tile_size.num_cols(FlashIdent::Query),
            layout: MatrixLayout::RowMajor,
        };

        tile
    }
}

#[cube]
impl<AP: AttentionPrecision, G: GlobalAttentionConfig> DummyKeyLoader<AP, G> {
    pub fn new(key: View<Line<AP::EI>, Coords3d>, #[comptime] config: G) -> Self {
        let tensor_reader =
            TensorReader::new(key, (0u32.runtime(), 0u32.runtime(), 0u32.runtime()));
        let stage_memory = StageMemory::new::<AttentionStageMemoryConfig>(
            1u32,
            StageIdent::Rhs,
            config.score_stage_memory_config(),
        );

        DummyKeyLoader::<AP, G> {
            tensor_reader,
            stage_memory,
            _phantom: PhantomData,
        }
    }

    pub fn reader(&self) -> FullStageToTileReader<AP::ES, AttentionTilingLayout> {
        FullStageToTileReader::<AP::ES, AttentionTilingLayout> {
            stage_memory: self.stage_memory,
            stage_ident: StageIdent::Rhs,
        }
    }

    pub fn load_transposed(&mut self, #[comptime] config: G) {
        // TODO this loader is bad, it's hardcoded to tile size (not stage) and is not coalesced

        comment!("Loading Key");
        let memory_config = config.global_memory_config(FlashIdent::Key);
        let mut slice = self.stage_memory.as_slice_mut(1u32);

        let tile_config = config.stage_config().tile_config();
        let num_rows = tile_config.attention_tile_size().num_rows(FlashIdent::Key);
        let num_cols = tile_config.attention_tile_size().num_cols(FlashIdent::Key);
        let num_units_per_row = tile_config.num_units_per_row(FlashIdent::Key);
        let num_cols_per_unit = tile_config.num_cols_per_unit(FlashIdent::Key);

        let num_tiles_row = config.tiling_scheme().partition_size.seq_kv;
        let num_tiles_col = config.tiling_scheme().partition_size.head_dim;

        let row = UNIT_POS_X / num_units_per_row;
        let col_start = (UNIT_POS_X % num_units_per_row) * num_cols_per_unit;

        // Assumes row tiling order
        let num_elements_per_tile = num_rows * num_cols;
        let tile_row_stride = num_tiles_col * num_elements_per_tile;
        let tile_col_stride = num_elements_per_tile;

        #[unroll]
        for tile_row in 0..num_tiles_row {
            #[unroll]
            for tile_col in 0..num_tiles_col {
                if row < num_rows {
                    #[unroll]
                    for i in 0..num_cols_per_unit {
                        let col = col_start + i;

                        if col < num_cols {
                            let index_load = row * num_cols + col;
                            let index_store = col * num_rows + row;
                            slice[index_store
                                + tile_row * tile_row_stride
                                + tile_col * tile_col_stride] =
                                Line::cast_from(self.tensor_reader.load_coalesced_in_tile(
                                    tile_row,
                                    tile_col,
                                    index_load,
                                    memory_config,
                                ));
                        }
                    }
                }
            }
        }
    }

    pub fn advance_view(&mut self, offset: u32) {
        self.tensor_reader
            .update_view(offset, comptime!(ViewDirection::Row));
    }
}

#[cube]
impl<AP: AttentionPrecision, G: GlobalAttentionConfig> DummyValueLoader<AP, G> {
    pub fn new(value: View<Line<AP::EI>, Coords3d>, #[comptime] config: G) -> Self {
        let tensor_reader =
            TensorReader::new(value, (0u32.runtime(), 0u32.runtime(), 0u32.runtime()));
        let stage_memory = StageMemory::new::<AttentionStageMemoryConfig>(
            1u32,
            StageIdent::Rhs,
            config.value_stage_memory_config(),
        );

        DummyValueLoader::<AP, G> {
            tensor_reader,
            stage_memory,
            _phantom: PhantomData,
        }
    }

    pub fn reader(&self) -> FullStageToTileReader<AP::ES, AttentionTilingLayout> {
        FullStageToTileReader::<AP::ES, AttentionTilingLayout> {
            stage_memory: self.stage_memory,
            stage_ident: StageIdent::Rhs,
        }
    }

    pub fn load(&mut self, #[comptime] config: G) {
        // TODO this loader is bad, it's hardcoded to tile size (not stage) and is not coalesced

        comment!("Loading Value");
        let memory_config = config.global_memory_config(FlashIdent::Value);
        let mut slice = self.stage_memory.as_slice_mut(1u32);

        let tile_config = config.stage_config().tile_config();
        let num_rows = tile_config
            .attention_tile_size()
            .num_rows(FlashIdent::Value);
        let num_cols = tile_config
            .attention_tile_size()
            .num_cols(FlashIdent::Value);
        let num_units_per_row = tile_config.num_units_per_row(FlashIdent::Value);
        let num_cols_per_unit = tile_config.num_cols_per_unit(FlashIdent::Value);

        let row = UNIT_POS_X / num_units_per_row;
        let col_start = (UNIT_POS_X % num_units_per_row) * num_cols_per_unit;

        if row < num_rows {
            #[unroll]
            for i in 0..num_cols_per_unit {
                let col = col_start + i;

                if col < num_cols {
                    let index = row * num_cols + col;
                    slice[index] = Line::cast_from(self.tensor_reader.load_coalesced_in_tile(
                        0u32,
                        0u32,
                        index,
                        memory_config,
                    ));
                }
            }
        }
    }

    pub fn advance_view(&mut self, offset: u32) {
        self.tensor_reader
            .update_view(offset, comptime!(ViewDirection::Row));
    }
}
