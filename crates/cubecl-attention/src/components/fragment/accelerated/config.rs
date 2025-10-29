use std::fmt::Debug;
use std::hash::Hash;

use crate::components::fragment::FragmentAttentionConfig;
use crate::components::{
    AttentionIdent, AttentionPrecision, AttentionSetupError, AttentionTileSize,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct AcceleratedFragmentAttentionConfig {
    plane_dim: u32,
    num_planes: u32,
    attention_tile_size: AttentionTileSize,
    query_stage_line_size: u32,
    key_value_stage_line_size: u32,
    check_bounds: bool,
}

impl FragmentAttentionConfig for AcceleratedFragmentAttentionConfig {
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

    fn check_bounds(&self) -> bool {
        self.check_bounds
    }

    fn num_rows_per_unit(&self) -> u32 {
        todo!()
    }

    fn causal_mask(&self) -> bool {
        todo!()
    }

    fn materialized_mask(&self) -> bool {
        todo!()
    }
}

impl AcceleratedFragmentAttentionConfig {
    pub fn new<AP: AttentionPrecision>(
        plane_dim: u32,
        attention_tile_size: AttentionTileSize,
        query_stage_line_size: u32,
        key_value_stage_line_size: u32,
        check_bounds: bool,
        num_planes: u32,
    ) -> Result<Self, AttentionSetupError> {
        Self {
            plane_dim,
            num_planes,
            attention_tile_size,
            query_stage_line_size,
            key_value_stage_line_size,
            check_bounds,
        }
        .validate()
    }

    pub fn validate(self) -> Result<Self, AttentionSetupError> {
        Ok(self)
    }
}
