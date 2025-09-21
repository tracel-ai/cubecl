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
use crate::components::tile::AttentionTilingLayout;
use crate::components::tile::dummy::{FlashMatmul, FlashMatmulConfig, FlashPrecision};
use crate::components::{AttentionPrecision, FlashIdent};

#[derive(CubeType)]
pub struct DummyQueryReader<AP: AttentionPrecision, G: GlobalAttentionConfig> {
    query: View<Line<AP::EI>, Coords2d>,

    #[cube(comptime)]
    _phantom: PhantomData<G>,
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
impl<AP: AttentionPrecision, G: GlobalAttentionConfig> DummyQueryReader<AP, G> {
    pub fn new(q_offset: u32, query: View<Line<AP::EI>, Coords2d>, #[comptime] config: G) -> Self {
        let attention_tile_size = config.stage_config().tile_config().attention_tile_size();
        let offset = (q_offset, 0);
        let size = (1u32, attention_tile_size.query_size()).runtime();
        let query = query.slice(offset, size);

        DummyQueryReader::<AP, G> {
            query,
            _phantom: PhantomData,
        }
    }

    pub fn stage_reader(&self, #[comptime] config: G) -> QueryRegisterReader<AP::EI> {
        comment!("Loading Query");

        let attention_tile_size = config.stage_config().tile_config().attention_tile_size();
        let tile = Tile::<AP::EI> {
            slice: self.query.to_linear_slice(),
            stride: attention_tile_size.num_cols(FlashIdent::Query),
            layout: MatrixLayout::RowMajor,
        };

        QueryRegisterReader::<AP::EI> { tile }
    }
}

#[cube]
impl<AP: AttentionPrecision, G: GlobalAttentionConfig> DummyKeyReader<AP, G> {
    pub fn new(key: View<Line<AP::EI>, Coords2d>, k_step: u32, #[comptime] config: G) -> Self {
        let global_iter = GlobalIterator::new(key, k_step, ViewDirection::Row, false);
        let stage_memory = StageMemory::new::<G::ScoreStageMemoryConfig>(
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
        // TODO this reader is bad, it's hardcoded to tile size (not stage) and is not coalesced

        comment!("Reading Key");
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
            let layout = TiledLayout::new(memory_config);
            let view = self.global_iter.view().view(layout);

            #[unroll]
            for i in 0..num_cols_per_unit {
                let col = col_start + i;

                if col < num_cols {
                    let index_load = row * num_cols + col;
                    let index_store = col * num_rows + row;
                    slice[index_store] =
                        Line::cast_from(view.read_checked(((0u32, 0u32).runtime(), index_load)));
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
    pub fn new(value: View<Line<AP::EI>, Coords2d>, k_step: u32, #[comptime] config: G) -> Self {
        let global_iter = GlobalIterator::new(value, k_step, ViewDirection::Row, false);
        let stage_memory = StageMemory::new::<G::ValueStageMemoryConfig>(
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
        // TODO this reader is bad, it's hardcoded to tile size (not stage) and is not coalesced

        comment!("Reading Value");
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
            let layout = TiledLayout::new(memory_config);
            let view = self.global_iter.view().view(layout);

            #[unroll]
            for i in 0..num_cols_per_unit {
                let col = col_start + i;

                if col < num_cols {
                    let index = row * num_cols + col;
                    slice[index] =
                        Line::cast_from(view.read_checked(((0u32, 0u32).runtime(), index)));
                }
            }
        }
    }

    pub fn advance_view(&mut self) {
        self.global_iter.advance();
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
