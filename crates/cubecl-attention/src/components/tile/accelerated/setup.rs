use cubecl_matmul::components::ComputeResources;

use crate::components::AttentionTileSize;
use crate::components::tile::SharedTileAttentionConfig;
use crate::components::tile::TileAttentionConfig;
use crate::components::tile::accelerated::BlackboxAcceleratedTileAttention;
use crate::components::tile::accelerated::local_tile::InnerLayout;
use crate::components::{
    AttentionBlueprint, AttentionPrecision, AttentionSetupError, InvalidConfigError,
    tile::TileAttentionFamily,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct BlackboxAcceleratedAttentionMatmulConfig {
    pub shared: SharedTileAttentionConfig,
    pub inner_layout: InnerLayout,
}

impl TileAttentionConfig for BlackboxAcceleratedAttentionMatmulConfig {
    fn plane_dim(&self) -> u32 {
        self.shared.plane_dim
    }

    fn num_planes(&self) -> u32 {
        self.shared.num_planes
    }

    fn attention_tile_size(&self) -> AttentionTileSize {
        self.shared.attention_tile_size
    }

    fn num_rows_per_unit(&self) -> u32 {
        match self.inner_layout {
            InnerLayout::Contiguous => 1u32,
            InnerLayout::SplitRows => 2u32,
        }
    }

    fn causal_mask(&self) -> bool {
        self.shared.causal_mask
    }

    fn materialized_mask(&self) -> bool {
        self.shared.materialized_mask
    }
}

impl TileAttentionFamily for BlackboxAcceleratedTileAttention {
    type TileAttention<F: AttentionPrecision> = BlackboxAcceleratedTileAttention;

    type Config = BlackboxAcceleratedAttentionMatmulConfig;

    fn requires_accelerator() -> bool {
        false
    }

    fn computation_resources() -> Result<ComputeResources, InvalidConfigError> {
        Ok(ComputeResources::Planes(1))
    }

    fn expand_blueprint(
        blueprint: &AttentionBlueprint,
    ) -> Result<Self::Config, AttentionSetupError> {
        validate(
            BlackboxAcceleratedAttentionMatmulConfig {
                shared: SharedTileAttentionConfig {
                    plane_dim: blueprint.plane_dim,
                    num_planes: blueprint.tiling_scheme.stage_size.seq_q,
                    attention_tile_size: blueprint.tiling_scheme.tile_size,
                    causal_mask: blueprint.causal,
                    materialized_mask: blueprint.masked,
                },
                inner_layout: if blueprint.two_rows_in_array_tile {
                    InnerLayout::SplitRows
                } else {
                    InnerLayout::Contiguous
                },
            },
            blueprint.reuse_key_value,
            blueprint.line_sizes.mask,
        )
    }
}

fn validate(
    config: BlackboxAcceleratedAttentionMatmulConfig,
    reuse_key_value: bool,
    line_sizes_mask: u8,
) -> Result<BlackboxAcceleratedAttentionMatmulConfig, AttentionSetupError> {
    if line_sizes_mask > 1 {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "Line size mask > 1 not supported yet on accelerated tile attention",
        )));
    }

    let softmax_num_rows = config.shared.attention_tile_size.seq_q;
    let softmax_num_cols = config.shared.attention_tile_size.seq_kv;
    let softmax_total = softmax_num_rows * softmax_num_cols;

    if softmax_total % config.shared.plane_dim != 0 {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "Softmax size should be divisible by plane dim",
        )));
    }

    if config.inner_layout == InnerLayout::Contiguous && softmax_num_rows > config.shared.plane_dim
    {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "More than one row per unit not supported with this inner layout",
        )));
    }

    if config.inner_layout == InnerLayout::SplitRows
        && softmax_total % (2 * config.shared.plane_dim) != 0
    {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "With split rows, units must have two elements each",
        )));
    }

    if config.shared.attention_tile_size.head_dim < config.shared.attention_tile_size.val_dim {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "Can't have tile head_dim < tile val dim (not sure why)",
        )));
    }

    if reuse_key_value {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "Can't reuse key/value because the fragment is col major for key and row major for value",
        )));
    }

    Ok(config)
}
