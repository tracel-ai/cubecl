use cubecl_matmul::components::ComputeResources;

use crate::components::{
    tile::dummy::{
        accelerated::{AcceleratedFlashMatmul, AcceleratedFlashMatmulConfig}, FlashMatmulFamily, FlashPrecision
    }, AttentionSetupError, InvalidConfigError
};

impl FlashMatmulFamily for AcceleratedFlashMatmul {
    type Matmul<F: FlashPrecision> = AcceleratedFlashMatmul;

    type Config = AcceleratedFlashMatmulConfig;

    fn setup<R: cubecl_core::Runtime>(
        client: &cubecl_core::prelude::ComputeClient<R::Server, R::Channel>,
        problem: &crate::components::AttentionProblem,
        selection: &crate::components::AttentionSelection,
        line_sizes: &crate::components::AttentionLineSizes,
    ) -> Result<Self::Config, AttentionSetupError> {
        AcceleratedFlashMatmulConfig::new(
            selection.plane_dim,
            (8, 8, 8).into(),
            1,
            line_sizes.query as u32,
            line_sizes.key as u32,
        )
    }

    fn requires_accelerator() -> bool {
        true
    }

    fn computation_resources() -> Result<ComputeResources, InvalidConfigError> {
        Ok(ComputeResources::Planes(1))
    }
}
