use std::fmt::Debug;
use std::hash::Hash;

use crate::components::attention_types::*;
use crate::components::fragment::AttentionMatmulConfig;
use crate::components::fragment::dummy_register::InnerLayout;
use crate::components::{
    AttentionIdent, AttentionPrecision, AttentionSetupError, AttentionTileSize,
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
    inner_layout: InnerLayout,
    causal_mask: bool,
    materialized_mask: bool,
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

    fn check_bounds(&self) -> bool {
        self.check_bounds
    }

    fn num_rows_per_unit(&self) -> u32 {
        match self.inner_layout {
            InnerLayout::Contiguous => 1u32,
            InnerLayout::SplitRows => 2u32,
        }
    }

    fn causal_mask(&self) -> bool {
        self.causal_mask
    }

    fn materialized_mask(&self) -> bool {
        self.materialized_mask
    }
}

impl DummyRegisterAttentionMatmulConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn new<AP: AttentionPrecision>(
        plane_dim: u32,
        attention_tile_size: AttentionTileSize,
        num_planes: u32,
        query_stage_line_size: u32,
        key_value_stage_line_size: u32,
        check_bounds: bool,
        two_rows_in_array_tile: bool,
        causal_mask: bool,
        materialized_mask: bool,
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
            inner_layout: if two_rows_in_array_tile {
                InnerLayout::SplitRows
            } else {
                InnerLayout::Contiguous
            },
            causal_mask,
            materialized_mask,
        }
        .validate()
    }

    pub fn validate(self) -> Result<Self, AttentionSetupError> {
        let softmax_num_rows = self.attention_tile_size.seq_q;
        let softmax_num_cols = self.attention_tile_size.seq_kv;
        let softmax_total = softmax_num_rows * softmax_num_cols;

        if softmax_total % self.plane_dim != 0 {
            return Err(AttentionSetupError::InvalidConfig(Box::new(
                "Softmax size should be divisible by plane dim",
            )));
        }

        if self.inner_layout == InnerLayout::Contiguous && softmax_num_rows > self.plane_dim {
            return Err(AttentionSetupError::InvalidConfig(Box::new(
                "More than one row per unit not supported with this inner layout",
            )));
        }

        if self.inner_layout == InnerLayout::SplitRows && softmax_total % (2 * self.plane_dim) != 0
        {
            return Err(AttentionSetupError::InvalidConfig(Box::new(
                "With split rows, units must have two elements each",
            )));
        }

        if self.attention_tile_size.head_dim < self.attention_tile_size.val_dim {
            return Err(AttentionSetupError::InvalidConfig(Box::new(
                "Can't have tile head_dim < tile val dim (not sure why)",
            )));
        }
        Ok(self)
    }

    pub fn inner_layout(&self) -> InnerLayout {
        self.inner_layout
    }
}
