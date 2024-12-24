use cubecl_core::prelude::*;

use crate::matmul::components::stage::{self, CommonStageInput};
use crate::matmul::components::{batch, global, tile, MatmulPrecision};
use crate::matmul::components::{MatmulConfigFactory, MatmulProblem};
use crate::matmul::kernels::matmul::AdvancedConfig;
use crate::matmul::kernels::{MatmulAvailabilityError, MatmulInvalidProblem};

/// Specifications for a matmul algorithm
pub trait Algorithm {
    type TileMatmul: tile::TileMatmulFamily;
    type StageMatmul: stage::StageMatmulFamily<Input = CommonStageInput<Self::TileMatmul>>;
    type GlobalMatmul: global::GlobalMatmulFamily;
    type BatchMatmul: batch::BatchMatmulFamily<Input = CommonStageInput<Self::TileMatmul>>;
    type Selection;

    fn cube_dim(selection: &Self::Selection) -> CubeDim;
    fn cube_count(selection: &Self::Selection, problem: &MatmulProblem) -> CubeCount;
    #[allow(clippy::type_complexity)]
    fn make_config(
        input: <Self::BatchMatmul as MatmulConfigFactory>::Input,
        problem: &MatmulProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        advanced_config: &AdvancedConfig,
    ) -> Result<<Self::BatchMatmul as MatmulConfigFactory>::Config, MatmulInvalidProblem> {
        let config =
            Self::BatchMatmul::make_config(input, problem, cube_dim, cube_count, advanced_config);
        problem.check_config(&config)?;
        Ok(config)
    }

    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &<Self::BatchMatmul as MatmulConfigFactory>::Config,
    ) -> Result<(), MatmulAvailabilityError> {
        Self::BatchMatmul::check_availability::<R, MP>(client, config)
    }

    fn advanced_config() -> AdvancedConfig {
        AdvancedConfig::default()
    }
}
