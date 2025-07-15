use crate::components::error::MatmulSetupError;
use crate::components::resource::ComputeResources;
use crate::components::tile::TileMatmulFamily;
use crate::components::tile::accelerated::config::AcceleratedConfig;
use crate::components::tile::accelerated::matmul::AcceleratedMatmul;
use crate::components::{
    InvalidConfigError, MatmulLineSizes, MatmulPrecision, MatmulProblem, MatmulSelection,
};
use cubecl_core::prelude::*;

impl TileMatmulFamily for AcceleratedMatmul {
    type Matmul<MP: MatmulPrecision> = AcceleratedMatmul;
    type Config = AcceleratedConfig;

    fn requires_accelerator() -> bool {
        true
    }

    fn computation_resources() -> Result<ComputeResources, InvalidConfigError> {
        Ok(ComputeResources::Planes(1))
    }

    fn setup<MP: MatmulPrecision, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        matmul_line_sizes: &MatmulLineSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        AcceleratedConfig::new::<MP, R>(
            client,
            selection.tiling_scheme,
            selection.plane_dim,
            problem.lhs_layout,
            problem.rhs_layout,
            matmul_line_sizes.lhs as u32,
            matmul_line_sizes.rhs as u32,
            matmul_line_sizes.out as u32,
            matmul_line_sizes.lhs as u32,
            matmul_line_sizes.rhs as u32,
        )
    }
}
