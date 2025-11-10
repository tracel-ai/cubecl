use cubecl_core::client::ComputeClient;
use cubecl_matmul::components::ComputeResources;

use crate::components::tile::accelerated::BlackboxAcceleratedAttentionMatmulConfig;
use crate::components::tile::accelerated::BlackboxAcceleratedTileAttention;
use crate::components::{
    AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError, InvalidConfigError, tile::TileAttentionFamily,
};

impl TileAttentionFamily for BlackboxAcceleratedTileAttention {
    type TileAttention<F: AttentionPrecision> = BlackboxAcceleratedTileAttention;

    type Config = BlackboxAcceleratedAttentionMatmulConfig;

    fn requires_accelerator() -> bool {
        false
    }

    fn computation_resources() -> Result<ComputeResources, InvalidConfigError> {
        Ok(ComputeResources::Planes(1))
    }

    fn setup<AP: AttentionPrecision, R: cubecl_core::Runtime>(
        _client: &ComputeClient<R::Server>,
        problem: &AttentionProblem,
        selection: &AttentionSelection,
        line_sizes: &AttentionLineSizes,
        num_planes: u32,
    ) -> Result<Self::Config, AttentionSetupError> {
        BlackboxAcceleratedAttentionMatmulConfig::new::<AP>(
            selection.plane_dim,
            selection.tiling_scheme.tile_size,
            num_planes,
            line_sizes.query as u32,
            line_sizes.key as u32,
            selection.two_rows_in_array_tile,
            selection.reuse_key_value,
            problem.causal,
            problem.masked,
        )
    }
}
