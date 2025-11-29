use cubecl_core::client::ComputeClient;
use cubecl_matmul::components::ComputeResources;

use crate::components::tile::unit_register::UnitRegisterTileAttention;
use crate::components::tile::{SharedTileAttentionConfig, TileAttentionConfig};
use crate::components::{AttentionElems, AttentionTileSize};
use crate::components::{
    AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError, InvalidConfigError, tile::TileAttentionFamily,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct UnitTileAttentionConfig {
    pub shared: SharedTileAttentionConfig,
}

impl TileAttentionConfig for UnitTileAttentionConfig {
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
        self.shared.attention_tile_size.seq_q
    }

    fn causal_mask(&self) -> bool {
        self.shared.causal_mask
    }

    fn materialized_mask(&self) -> bool {
        self.shared.materialized_mask
    }
}

impl TileAttentionFamily for UnitRegisterTileAttention {
    type TileAttention<F: AttentionPrecision> = UnitRegisterTileAttention;

    type Config = UnitTileAttentionConfig;

    fn requires_accelerator() -> bool {
        false
    }

    fn computation_resources() -> Result<ComputeResources, InvalidConfigError> {
        Ok(ComputeResources::Units(1))
    }

    fn setup<R: cubecl_core::Runtime>(
        _client: &ComputeClient<R>,
        problem: &AttentionProblem,
        selection: &AttentionSelection,
        _line_sizes: &AttentionLineSizes,
        num_planes: u32,
        _dtypes: &AttentionElems,
    ) -> Result<Self::Config, AttentionSetupError> {
        Ok(UnitTileAttentionConfig {
            shared: SharedTileAttentionConfig {
                plane_dim: selection.plane_dim,
                attention_tile_size: selection.tiling_scheme.tile_size,
                num_planes,
                causal_mask: problem.causal,
                materialized_mask: problem.masked,
            },
        })
    }
}
