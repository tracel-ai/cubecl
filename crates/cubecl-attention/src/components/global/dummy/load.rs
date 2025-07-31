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

#[derive(CubeType)]
pub struct DummyQueryLoader<AP: AttentionPrecision> {
    tensor_reader: TensorReader<AP::EI>,

    #[cube(comptime)]
    _phantom: PhantomData<AP>,
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

        DummyQueryLoader::<AP> {
            tensor_reader,
            _phantom: PhantomData,
        }
    }

    pub fn load(&self) -> QueryRegisterReader<AP::ES> {
        comment!("Loading Query");
        QueryRegisterReader::<AP::ES> {
            row: Array::vectorized(8u32, 1u32),
            fragment: cmma::Matrix::from_slice(
                cmma::MatrixIdent::A,
                8,
                8,
                8,
                cmma::MatrixLayout::RowMajor,
                &self
                    .tensor_reader
                    .tensor
                    .as_slice(0, 64)
                    .try_cast_unchecked(),
                8,
            ),
        }
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

    pub fn load(&mut self) {
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
pub struct QueryRegisterReader<E: Float> {
    row: Array<Line<E>>,
    fragment: cmma::Matrix<E>,
}

#[cube]
impl<E: Float> QueryRegisterReader<E> {
    pub fn row(&self) -> &Array<Line<E>> {
        &self.row
    }
    pub fn fragment(&self) -> &cmma::Matrix<E> {
        &self.fragment
    }
}
