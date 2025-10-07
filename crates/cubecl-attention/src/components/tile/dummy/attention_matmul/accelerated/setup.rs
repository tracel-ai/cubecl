use cubecl_matmul::components::ComputeResources;

use crate::components::{
    AttentionPrecision, AttentionSetupError, InvalidConfigError,
    tile::dummy::{
        AttentionMatmulFamily,
        accelerated::{AcceleratedAttentionMatmul, AcceleratedAttentionMatmulConfig},
    },
};

impl AttentionMatmulFamily for AcceleratedAttentionMatmul {
    type Matmul<AP: AttentionPrecision> = AcceleratedAttentionMatmul;

    type Config = AcceleratedAttentionMatmulConfig;

    fn requires_accelerator() -> bool {
        true
    }

    fn computation_resources() -> Result<ComputeResources, InvalidConfigError> {
        Ok(ComputeResources::Planes(1))
    }

    fn setup<AP: AttentionPrecision, R: cubecl_core::Runtime>(
        _client: &cubecl_core::prelude::ComputeClient<R::Server, R::Channel>,
        problem: &crate::components::AttentionProblem,
        selection: &crate::components::AttentionSelection,
        line_sizes: &crate::components::AttentionLineSizes,
        num_planes: u32,
    ) -> Result<Self::Config, AttentionSetupError> {
        AcceleratedAttentionMatmulConfig::new::<AP>(
            selection.plane_dim,
            selection.tiling_scheme.tile_size,
            num_planes,
            line_sizes.query as u32,
            line_sizes.key as u32,
            !(problem.seq_kv as u32).is_multiple_of(selection.tiling_scheme.tile_size.seq_kv),
        )
    }
}
