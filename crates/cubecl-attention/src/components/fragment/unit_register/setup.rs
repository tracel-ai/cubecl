use cubecl_core::client::ComputeClient;
use cubecl_matmul::components::ComputeResources;

use crate::components::fragment::unit_register::UnitRegisterAttentionMatmul;
use crate::components::fragment::unit_register::UnitRegisterAttentionMatmulConfig;
use crate::components::{
    AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError, InvalidConfigError, fragment::AttentionMatmulFamily,
};

impl AttentionMatmulFamily for UnitRegisterAttentionMatmul {
    type Matmul<F: AttentionPrecision> = UnitRegisterAttentionMatmul;

    type Config = UnitRegisterAttentionMatmulConfig;

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
        todo!()
    }
}
