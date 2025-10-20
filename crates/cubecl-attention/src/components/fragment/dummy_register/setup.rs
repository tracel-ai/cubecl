use cubecl_core::client::ComputeClient;
use cubecl_matmul::components::ComputeResources;

use crate::components::fragment::dummy_register::DummyRegisterAttentionMatmul;
use crate::components::fragment::dummy_register::DummyRegisterAttentionMatmulConfig;
use crate::components::{
    AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError, InvalidConfigError, fragment::AttentionMatmulFamily,
};

impl AttentionMatmulFamily for DummyRegisterAttentionMatmul {
    type Matmul<F: AttentionPrecision> = DummyRegisterAttentionMatmul;

    type Config = DummyRegisterAttentionMatmulConfig;

    fn requires_accelerator() -> bool {
        true
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
        DummyRegisterAttentionMatmulConfig::new::<AP>(
            selection.plane_dim,
            selection.tiling_scheme.tile_size,
            num_planes,
            line_sizes.query as u32,
            line_sizes.key as u32,
            !(problem.seq_kv as u32).is_multiple_of(selection.tiling_scheme.tile_size.seq_kv),
            selection.two_rows_in_array_tile,
            problem.causal,
            problem.masked,
        )
    }
}
