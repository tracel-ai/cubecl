use cubecl_core::client::ComputeClient;
use cubecl_matmul::components::ComputeResources;

use crate::components::fragment::unit_register::UnitRegisterFragmentAttention;
use crate::components::fragment::unit_register::UnitRegisterFragmentAttentionConfig;
use crate::components::{
    AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError, InvalidConfigError, fragment::FragmentAttentionFamily,
};

impl FragmentAttentionFamily for UnitRegisterFragmentAttention {
    type FragmentAttention<F: AttentionPrecision> = UnitRegisterFragmentAttention;

    type Config = UnitRegisterFragmentAttentionConfig;

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
        UnitRegisterFragmentAttentionConfig::new::<AP>(
            selection.plane_dim,
            selection.tiling_scheme.tile_size,
            line_sizes.query as u32,
            line_sizes.key as u32,
            !(problem.seq_kv as u32).is_multiple_of(selection.tiling_scheme.tile_size.seq_kv),
            num_planes,
            problem.causal,
            problem.masked,
        )
    }
}
