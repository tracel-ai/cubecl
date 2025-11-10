use std::fmt::Debug;
use std::hash::Hash;

use crate::components::tile::TileAttentionConfig;
use crate::components::{AttentionPrecision, AttentionSetupError, AttentionTileSize};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct UnitRegisterTileAttentionConfig {
    plane_dim: u32,
    num_planes: u32,
    attention_tile_size: AttentionTileSize,
    query_stage_line_size: u32,
    key_value_stage_line_size: u32,
    causal_mask: bool,
    materialized_mask: bool,
}

impl TileAttentionConfig for UnitRegisterTileAttentionConfig {
    fn plane_dim(&self) -> u32 {
        self.plane_dim
    }

    fn num_planes(&self) -> u32 {
        self.num_planes
    }

    fn attention_tile_size(&self) -> AttentionTileSize {
        self.attention_tile_size
    }

    fn num_rows_per_unit(&self) -> u32 {
        self.attention_tile_size.seq_q
    }

    fn causal_mask(&self) -> bool {
        self.causal_mask
    }

    fn materialized_mask(&self) -> bool {
        self.materialized_mask
    }
}

impl UnitRegisterTileAttentionConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn new<AP: AttentionPrecision>(
        plane_dim: u32,
        attention_tile_size: AttentionTileSize,
        query_stage_line_size: u32,
        key_value_stage_line_size: u32,
        num_planes: u32,
        causal_mask: bool,
        materialized_mask: bool,
    ) -> Result<Self, AttentionSetupError> {
        Self {
            plane_dim,
            num_planes,
            attention_tile_size,
            query_stage_line_size,
            key_value_stage_line_size,
            causal_mask,
            materialized_mask,
        }
        .validate()
    }

    pub fn validate(self) -> Result<Self, AttentionSetupError> {
        Ok(self)
    }
}
