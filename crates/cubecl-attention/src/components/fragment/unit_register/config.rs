use std::fmt::Debug;
use std::hash::Hash;

use crate::components::fragment::AttentionMatmulConfig;
use crate::components::{
    AttentionIdent, AttentionPrecision, AttentionSetupError, AttentionTileSize,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct UnitRegisterAttentionMatmulConfig {}

impl AttentionMatmulConfig for UnitRegisterAttentionMatmulConfig {
    fn plane_dim(&self) -> u32 {
        todo!()
    }

    fn num_planes(&self) -> u32 {
        todo!()
    }

    fn stage_line_size(&self, ident: AttentionIdent) -> u32 {
        todo!()
    }

    fn attention_tile_size(&self) -> AttentionTileSize {
        todo!()
    }

    fn cast_query(&self) -> bool {
        todo!()
    }

    fn check_bounds(&self) -> bool {
        todo!()
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

impl UnitRegisterAttentionMatmulConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn new<AP: AttentionPrecision>() -> Result<Self, AttentionSetupError> {
        Self {}.validate()
    }

    pub fn validate(self) -> Result<Self, AttentionSetupError> {
        Ok(self)
    }
}
