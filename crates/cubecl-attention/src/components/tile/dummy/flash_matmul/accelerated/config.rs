use cubecl_matmul::components::{MatrixLayout, StageIdent, TileSize, tile::TileConfig};
use std::fmt::Debug;
use std::hash::Hash;

use crate::components::{AttentionSetupError, FlashIdent, tile::dummy::FlashMatmulConfig};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct AcceleratedFlashMatmulConfig {
    plane_dim: u32,
    score_config: ScoreConfig,
    value_config: ValueConfig,
    tile_size: TileSize,
    num_planes: u32,
    query_stage_line_size: u32,
    key_value_stage_line_size: u32,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct ScoreConfig {
    plane_dim: u32,
    tile_size: TileSize,
    query_stage_line_size: u32,
    key_value_stage_line_size: u32,
}
impl TileConfig for ScoreConfig {
    fn plane_dim(&self) -> u32 {
        self.plane_dim
    }

    fn matrix_layout(&self, _ident: StageIdent) -> MatrixLayout {
        MatrixLayout::RowMajor
    }

    fn stage_line_size(&self, ident: StageIdent) -> u32 {
        match ident {
            StageIdent::Lhs => self.query_stage_line_size,
            StageIdent::Rhs => self.key_value_stage_line_size,
            StageIdent::Acc => todo!(),
        }
    }

    fn global_line_size(&self, ident: StageIdent) -> u32 {
        todo!()
    }

    fn tile_size(&self) -> &TileSize {
        &self.tile_size
    }
}
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct ValueConfig {
    plane_dim: u32,
    tile_size: TileSize,
    key_value_stage_line_size: u32,
}
impl TileConfig for ValueConfig {
    fn plane_dim(&self) -> u32 {
        self.plane_dim
    }

    fn matrix_layout(&self, _ident: StageIdent) -> MatrixLayout {
        MatrixLayout::RowMajor
    }

    fn stage_line_size(&self, ident: StageIdent) -> u32 {
        match ident {
            StageIdent::Lhs => todo!(),
            StageIdent::Rhs => self.key_value_stage_line_size,
            StageIdent::Acc => todo!(),
        }
    }

    fn global_line_size(&self, ident: StageIdent) -> u32 {
        todo!()
    }

    fn tile_size(&self) -> &TileSize {
        &self.tile_size
    }
}

impl FlashMatmulConfig for AcceleratedFlashMatmulConfig {
    type ScoreConfig = ScoreConfig;
    type ValueConfig = ValueConfig;

    fn score_config(&self) -> Self::ScoreConfig {
        self.score_config
    }

    fn value_config(&self) -> Self::ValueConfig {
        self.value_config
    }

    fn plane_dim(&self) -> u32 {
        self.plane_dim
    }

    fn num_planes(&self) -> u32 {
        self.num_planes
    }

    fn rows_per_plane(&self) -> u32 {
        1
    }

    fn reuse_key_value(&self) -> bool {
        true
    }

    fn stage_line_size(&self, ident: FlashIdent) -> u32 {
        match ident {
            FlashIdent::Query => self.query_stage_line_size,
            FlashIdent::Key => self.key_value_stage_line_size,
            FlashIdent::Value => self.key_value_stage_line_size,
            FlashIdent::Mask => todo!(),
            FlashIdent::Out => 1,
        }
    }

    fn tile_size(&self) -> TileSize {
        self.tile_size
    }
}

impl AcceleratedFlashMatmulConfig {
    pub fn new(
        plane_dim: u32,
        tile_size: TileSize,
        num_planes: u32,
        query_stage_line_size: u32,
        key_value_stage_line_size: u32,
    ) -> Result<Self, AttentionSetupError> {
        let score_config = ScoreConfig {
            plane_dim,
            tile_size,
            query_stage_line_size,
            key_value_stage_line_size,
        };
        let value_config = ValueConfig {
            plane_dim,
            tile_size,
            key_value_stage_line_size,
        };
        Self {
            plane_dim,
            score_config,
            value_config,
            tile_size,
            num_planes,
            query_stage_line_size,
            key_value_stage_line_size,
        }
        .validate()
    }

    pub fn validate(self) -> Result<Self, AttentionSetupError> {
        Ok(self)
    }
}
