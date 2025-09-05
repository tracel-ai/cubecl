use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::global::memory::TensorReader;
use cubecl_matmul::components::stage::{FullStageToTileReader, StageMemory};
use cubecl_matmul::components::tile::Tile;
use cubecl_matmul::components::{MatrixLayout, StageIdent};
use cubecl_std::tensor::{View, layout::Coords3d};
use std::marker::PhantomData;

use crate::components::global::base::GlobalAttentionConfig;
use crate::components::stage::StageAttentionConfig;
use crate::components::tile::AttentionTilingLayout;
use crate::components::tile::dummy::{FlashMatmul, FlashMatmulConfig, FlashPrecision};
use crate::components::{AttentionPrecision, FlashIdent};

#[derive(CubeType)]
pub struct DummyQueryLoader<AP: AttentionPrecision> {
    tensor_reader: TensorReader<AP::EI>,
}

#[derive(CubeType)]
pub struct DummyKeyLoader<AP: AttentionPrecision, G: GlobalAttentionConfig> {
    tensor_reader: TensorReader<AP::EI>,
    stage_memory: StageMemory<AP::ES, AttentionTilingLayout>,

    #[cube(comptime)]
    _phantom: PhantomData<(AP, G)>,
}
#[derive(CubeType)]
pub struct DummyValueLoader<AP: AttentionPrecision, G: GlobalAttentionConfig> {
    tensor_reader: TensorReader<AP::EI>,
    stage_memory: StageMemory<AP::ES, AttentionTilingLayout>,

    #[cube(comptime)]
    _phantom: PhantomData<(AP, G)>,
}

#[cube]
impl<AP: AttentionPrecision> DummyQueryLoader<AP> {
    pub fn new(query: View<Line<AP::EI>, Coords3d>) -> Self {
        let tensor_reader =
            TensorReader::new(query, (0u32.runtime(), 0u32.runtime(), 0u32.runtime()));

        DummyQueryLoader::<AP> { tensor_reader }
    }

    pub fn reader<G: GlobalAttentionConfig>(
        &self,
        #[comptime] config: G,
    ) -> QueryRegisterReader<AP::EI> {
        comment!("Loading Query");

        let attention_tile_size = config.stage_config().tile_config().attention_tile_size();
        let tile = Tile::<AP::EI> {
            slice: self.tensor_reader.view.slice(
                (0u32.runtime(), 0u32.runtime(), 0u32.runtime()),
                attention_tile_size.query_size(),
            ),
            stride: attention_tile_size.num_cols(FlashIdent::Query),
            layout: MatrixLayout::RowMajor,
        };

        QueryRegisterReader::<AP::EI> { tile }
    }
}

#[cube]
impl<AP: AttentionPrecision, G: GlobalAttentionConfig> DummyKeyLoader<AP, G> {
    pub fn new(key: View<Line<AP::EI>, Coords3d>, #[comptime] config: G) -> Self {
        let tensor_reader =
            TensorReader::new(key, (0u32.runtime(), 0u32.runtime(), 0u32.runtime()));
        let stage_memory = StageMemory::new::<G::ScoreStageMemoryConfig>(
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

        let row = UNIT_POS_X / num_units_per_row;
        let col_start = (UNIT_POS_X % num_units_per_row) * num_cols_per_unit;

        if row < num_rows {
            #[unroll]
            for i in 0..num_cols_per_unit {
                let col = col_start + i;

                if col < num_cols {
                    let index_load = row * num_cols + col;
                    let index_store = col * num_rows + row;
                    slice[index_store] =
                        Line::cast_from(self.tensor_reader.load_coalesced_in_tile(
                            0u32,
                            0u32,
                            index_load,
                            memory_config,
                        ));
                }
            }
        }
    }
}

#[cube]
impl<AP: AttentionPrecision, G: GlobalAttentionConfig> DummyValueLoader<AP, G> {
    pub fn new(value: View<Line<AP::EI>, Coords3d>, #[comptime] config: G) -> Self {
        let tensor_reader =
            TensorReader::new(value, (0u32.runtime(), 0u32.runtime(), 0u32.runtime()));
        let stage_memory = StageMemory::new::<G::ValueStageMemoryConfig>(
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
}

#[derive(CubeType)]
pub struct QueryRegisterReader<E: Numeric> {
    tile: Tile<E>,
}

#[cube]
impl<E: Numeric> QueryRegisterReader<E> {
    pub fn read_tile<FP: FlashPrecision, FM: FlashMatmul<FP>>(
        &self,
        #[comptime] config: FM::Config,
    ) -> FM::Query {
        FM::allocate_fill_query(&self.tile, config)
    }
}
