use cubecl_core::client::ComputeClient;
use cubecl_matmul::components::ComputeResources;

use crate::components::tile::unit_register::UnitRegisterTileAttention;
use crate::components::tile::unit_register::UnitRegisterTileAttentionConfig;
use crate::components::{
    AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError, InvalidConfigError, tile::TileAttentionFamily,
};

impl TileAttentionFamily for UnitRegisterTileAttention {
    type TileAttention<F: AttentionPrecision> = UnitRegisterTileAttention;

    type Config = UnitRegisterTileAttentionConfig;

    fn requires_accelerator() -> bool {
        false
    }

    fn computation_resources() -> Result<ComputeResources, InvalidConfigError> {
        Ok(ComputeResources::Units(1))
    }

    fn setup<AP: AttentionPrecision, R: cubecl_core::Runtime>(
        _client: &ComputeClient<R::Server>,
        problem: &AttentionProblem,
        selection: &AttentionSelection,
        line_sizes: &AttentionLineSizes,
        num_planes: u32,
    ) -> Result<Self::Config, AttentionSetupError> {
        UnitRegisterTileAttentionConfig::new::<AP>(
            selection.plane_dim,
            selection.tiling_scheme.tile_size,
            line_sizes.query as u32,
            line_sizes.key as u32,
            num_planes,
            problem.causal,
            problem.masked,
        )
    }
}
