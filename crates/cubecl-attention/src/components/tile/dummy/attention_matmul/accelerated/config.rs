use std::fmt::Debug;
use std::hash::Hash;

use crate::components::{
    AttentionIdent, AttentionPrecision, AttentionSetupError, AttentionTileSize, attention_types::*,
    tile::dummy::AttentionMatmulConfig,
};
use cubecl_core::frontend::CubePrimitive;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct AcceleratedAttentionMatmulConfig {
    plane_dim: u32,
    attention_tile_size: AttentionTileSize,
    num_planes: u32,
    query_stage_line_size: u32,
    key_value_stage_line_size: u32,
    cast_query: bool,
    check_bounds: bool,
}

impl AttentionMatmulConfig for AcceleratedAttentionMatmulConfig {
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

    fn check_bounds(&self) -> bool {
        self.check_bounds
    }

    fn num_rows_per_unit(&self) -> u32 {
        todo!()
    }
}

impl AcceleratedAttentionMatmulConfig {
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
        Ok(self)
    }
}
