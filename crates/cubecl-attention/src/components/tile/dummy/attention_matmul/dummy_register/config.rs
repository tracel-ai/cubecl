use cubecl_matmul::components::{MatrixLayout, StageIdent, TileSize, tile::TileConfig};
use std::fmt::Debug;
use std::hash::Hash;

use crate::components::attention_types::*;
use crate::components::{
    AttentionIdent, AttentionPrecision, AttentionSetupError, AttentionTileSize,
    tile::dummy::AttentionMatmulConfig,
};
use cubecl_core::frontend::CubePrimitive;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct DummyRegisterAttentionMatmulConfig {
    plane_dim: u32,
    attention_tile_size: AttentionTileSize,
    num_planes: u32,
    query_stage_line_size: u32,
    key_value_stage_line_size: u32,
    cast_query: bool,
    check_bounds: bool,
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
            StageIdent::Out => todo!(),
        }
    }

    fn global_line_size(&self, _ident: StageIdent) -> u32 {
        panic!()
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
            StageIdent::Out => todo!(),
        }
    }

    fn global_line_size(&self, _ident: StageIdent) -> u32 {
        panic!()
    }

    fn tile_size(&self) -> &TileSize {
        &self.tile_size
    }
}

impl AttentionMatmulConfig for DummyRegisterAttentionMatmulConfig {
    fn plane_dim(&self) -> u32 {
        self.plane_dim
    }

    fn num_planes(&self) -> u32 {
        self.num_planes
    }

    fn stage_line_size(&self, ident: AttentionIdent) -> u32 {
        match ident {
            AttentionIdent::Query => self.query_stage_line_size,
            AttentionIdent::Key => self.key_value_stage_line_size,
            AttentionIdent::Softmax => unreachable!("Not a materialized stage"),
            AttentionIdent::Value => self.key_value_stage_line_size,
            AttentionIdent::Mask => todo!(),
            AttentionIdent::Out => 1,
        }
    }

    fn attention_tile_size(&self) -> AttentionTileSize {
        self.attention_tile_size
    }

    fn cast_query(&self) -> bool {
        self.cast_query
    }

    fn num_units_per_row(&self, ident: AttentionIdent) -> u32 {
        self.plane_dim / self.attention_tile_size.num_rows(ident)
    }

    fn num_cols_per_unit(&self, ident: AttentionIdent) -> u32 {
        self.attention_tile_size
            .num_cols(ident)
            .div_ceil(self.num_units_per_row(ident))
    }

    fn num_rows_per_unit(&self, ident: AttentionIdent) -> u32 {
        self.attention_tile_size
            .num_rows(ident)
            .div_ceil(self.plane_dim)
    }

    fn check_bounds(&self) -> bool {
        self.check_bounds
    }
}

impl DummyRegisterAttentionMatmulConfig {
    pub fn new<AP: AttentionPrecision>(
        plane_dim: u32,
        attention_tile_size: AttentionTileSize,
        num_planes: u32,
        query_stage_line_size: u32,
        key_value_stage_line_size: u32,
        check_bounds: bool,
    ) -> Result<Self, AttentionSetupError> {
        Self {
            plane_dim,
            attention_tile_size,
            num_planes,
            query_stage_line_size,
            key_value_stage_line_size,
            cast_query: QG::<AP>::as_type_native_unchecked()
                == QT::<AP>::as_type_native_unchecked(),
            check_bounds,
        }
        .validate()
    }

    pub fn validate(self) -> Result<Self, AttentionSetupError> {
        if self.attention_tile_size.head_dim < self.attention_tile_size.val_dim {
            return Err(AttentionSetupError::InvalidConfig(Box::new(
                "Can't have tile head_dim < tile val dim (not sure why)",
            )));
        }
        Ok(self)
    }
}
