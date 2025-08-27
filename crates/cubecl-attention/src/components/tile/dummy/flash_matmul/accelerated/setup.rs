use cubecl_matmul::components::ComputeResources;

use crate::components::{
    AttentionPrecision, AttentionSetupError, InvalidConfigError,
    tile::dummy::{
        FlashMatmulFamily, FlashPrecision,
        accelerated::{AcceleratedFlashMatmul, AcceleratedFlashMatmulConfig},
    },
};

impl FlashMatmulFamily for AcceleratedFlashMatmul {
    type Matmul<F: FlashPrecision> = AcceleratedFlashMatmul;

    type Config = AcceleratedFlashMatmulConfig;

    fn requires_accelerator() -> bool {
        true
    }

    fn computation_resources() -> Result<ComputeResources, InvalidConfigError> {
        Ok(ComputeResources::Planes(1))
    }

    fn setup<AP: AttentionPrecision, R: cubecl_core::Runtime>(
        _client: &cubecl_core::prelude::ComputeClient<R::Server, R::Channel>,
        _problem: &crate::components::AttentionProblem,
        selection: &crate::components::AttentionSelection,
        line_sizes: &crate::components::AttentionLineSizes,
    ) -> Result<Self::Config, AttentionSetupError> {
        AcceleratedFlashMatmulConfig::new::<AP>(
            selection.plane_dim,
            selection.attention_tile_size,
            1,
            line_sizes.query as u32,
            line_sizes.key as u32,
        )
    }
}
