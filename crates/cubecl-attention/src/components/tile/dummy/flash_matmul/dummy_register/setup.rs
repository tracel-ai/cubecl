use cubecl_core::client::ComputeClient;
use cubecl_matmul::components::ComputeResources;

use crate::components::{
    AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError, InvalidConfigError,
    tile::dummy::{
        FlashMatmulFamily, FlashPrecision,
        dummy_register::{DummyRegisterFlashMatmul, DummyRegisterFlashMatmulConfig},
    },
};

impl FlashMatmulFamily for DummyRegisterFlashMatmul {
    type Matmul<F: FlashPrecision> = DummyRegisterFlashMatmul;

    type Config = DummyRegisterFlashMatmulConfig;

    fn requires_accelerator() -> bool {
        true
    }

    fn computation_resources() -> Result<ComputeResources, InvalidConfigError> {
        Ok(ComputeResources::Planes(1))
    }

    fn setup<AP: AttentionPrecision, R: cubecl_core::Runtime>(
        _client: &ComputeClient<R::Server, R::Channel>,
        _problem: &AttentionProblem,
        selection: &AttentionSelection,
        line_sizes: &AttentionLineSizes,
    ) -> Result<Self::Config, AttentionSetupError> {
        DummyRegisterFlashMatmulConfig::new::<AP>(
            selection.plane_dim,
            selection.attention_tile_size,
            1,
            line_sizes.query as u32,
            line_sizes.key as u32,
        )
    }
}
