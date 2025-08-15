use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::global::memory::{GlobalMemoryConfig, TensorReader};
use cubecl_matmul::components::stage::{FullStageToTileReader, StageMemory};
use cubecl_matmul::components::tile::Tile;
use cubecl_matmul::components::{MatrixLayout, StageIdent};
use cubecl_std::tensor::r#virtual::VirtualTensor;
use std::marker::PhantomData;

use crate::components::AttentionPrecision;
use crate::components::global::base::GlobalAttentionConfig;
use crate::components::stage::AttentionTilingLayout;
use crate::components::tile::ScoreMatmul;

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
    pub fn new(query: VirtualTensor<AP::EI>) -> Self {
        let tensor_reader = TensorReader::new(query, 0, 0, 0);

        DummyQueryLoader::<AP> { tensor_reader }
    }

    pub fn reader(&self) -> QueryRegisterReader<AP> {
        comment!("Loading Query");

        let tile = Tile::<AP::EI> {
            slice: self.tensor_reader.tensor.as_slice(0, 64),
            stride: 8,
            layout: MatrixLayout::RowMajor,
        };

        QueryRegisterReader::<AP> { tile }
    }
}

#[cube]
impl<AP: AttentionPrecision, G: GlobalAttentionConfig> DummyKeyLoader<AP, G> {
    pub fn new(key: VirtualTensor<AP::EI>, #[comptime] config: G) -> Self {
        let tensor_reader = TensorReader::new(key, 0, 0, 0);
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

    pub fn load_transposed(&mut self) {
        comment!("Loading Key");
        let config = comptime!(GlobalMemoryConfig {
            elements_in_tile_row: 8,
            elements_in_tile_col: 8,
            elements_in_stage_row: 8,
            elements_in_stage_col: 8,
            global_line_size: 1,
            check_row_bounds: false,
            check_col_bounds: false,
            matrix_layout: MatrixLayout::RowMajor,
        });

        let index_load_0 = UNIT_POS_X;
        let index_load_1 = UNIT_POS_X + 32;
        let (row_0, col_0) = (index_load_0 / 8, index_load_0 % 8);
        let (row_1, col_1) = (index_load_1 / 8, index_load_1 % 8);
        let index_store_0 = col_0 * 8 + row_0;
        let index_store_1 = col_1 * 8 + row_1;

        let line0 = self
            .tensor_reader
            .load_coalesced_in_tile(0u32, 0u32, index_load_0, config);
        let line1 = self
            .tensor_reader
            .load_coalesced_in_tile(0u32, 0u32, index_load_1, config);

        let mut slice = self.stage_memory.as_slice_mut(1u32);
        slice[index_store_0] = Line::cast_from(line0);
        slice[index_store_1] = Line::cast_from(line1);
    }
}

#[cube]
impl<AP: AttentionPrecision, G: GlobalAttentionConfig> DummyValueLoader<AP, G> {
    pub fn new(value: VirtualTensor<AP::EI>, #[comptime] config: G) -> Self {
        let tensor_reader = TensorReader::new(value, 0, 0, 0);
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

    pub fn load(&mut self) {
        comment!("Loading Value");
        let config = comptime!(GlobalMemoryConfig {
            elements_in_tile_row: 8,
            elements_in_tile_col: 8,
            elements_in_stage_row: 8,
            elements_in_stage_col: 8,
            global_line_size: 1,
            check_row_bounds: false,
            check_col_bounds: false,
            matrix_layout: MatrixLayout::RowMajor,
        });

        let line0 = self
            .tensor_reader
            .load_coalesced_in_tile(0u32, 0u32, UNIT_POS_X, config);
        let line1 = self
            .tensor_reader
            .load_coalesced_in_tile(0u32, 0u32, UNIT_POS_X + 32, config);

        let mut slice = self.stage_memory.as_slice_mut(1u32);
        slice[UNIT_POS_X] = Line::cast_from(line0);
        slice[UNIT_POS_X + 32] = Line::cast_from(line1);
    }
}

#[derive(CubeType)]
pub struct QueryRegisterReader<AP: AttentionPrecision> {
    tile: Tile<AP::EI>,
}

#[cube]
impl<AP: AttentionPrecision> QueryRegisterReader<AP> {
    pub fn read_tile<TM: ScoreMatmul<AP>>(&self, #[comptime] tile_config: TM::Config) -> TM::Lhs {
        let fragment = TM::allocate_fill_cast_lhs(&self.tile, tile_config);
        fragment
    }
}
