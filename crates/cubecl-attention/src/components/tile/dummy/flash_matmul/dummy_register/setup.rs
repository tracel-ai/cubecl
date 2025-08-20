use cubecl_matmul::components::ComputeResources;

use crate::components::{
    AttentionPrecision, AttentionSetupError, InvalidConfigError,
    tile::dummy::{
        AttentionTileSize, FlashMatmulFamily, FlashPrecision,
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
        _client: &cubecl_core::prelude::ComputeClient<R::Server, R::Channel>,
        _problem: &crate::components::AttentionProblem,
        selection: &crate::components::AttentionSelection,
        line_sizes: &crate::components::AttentionLineSizes,
    ) -> Result<Self::Config, AttentionSetupError> {
        DummyRegisterFlashMatmulConfig::new::<AP>(
            selection.plane_dim,
            AttentionTileSize {
                seq_q: 8,
                head_dim: 8,
                seq_kv: 8,
                val_dim: 8,
            },
            1,
            line_sizes.query as u32,
            line_sizes.key as u32,
        )
    }
}
