use cubecl_core::prelude::*;

use crate::matmul::components::{
    batch, global, stage, tile, CompleteStageTiling, MatmulConfigFactory, MatmulPrecision,
    MatmulProblem, MatmulSelection,
};
use crate::matmul::kernels::{MatmulAvailabilityError, MatmulLaunchError};

/// Specifications for a matmul algorithm
pub trait Algorithm {
    type TileMatmul: tile::TileMatmulFamily;
    type StageMatmul: stage::StageMatmulFamily<Input = CompleteStageTiling>;
    type GlobalMatmul: global::GlobalMatmulFamily;
    type BatchMatmul: batch::BatchMatmulFamily<Input = CompleteStageTiling>;

    fn cube_dim(selection: &MatmulSelection) -> CubeDim;
    fn cube_count(selection: &MatmulSelection, problem: &MatmulProblem) -> CubeCount;

    #[allow(clippy::type_complexity)]
    fn make_config(
        input: <Self::BatchMatmul as MatmulConfigFactory>::Input,
        problem: &MatmulProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        quantized: bool,
    ) -> Result<<Self::BatchMatmul as MatmulConfigFactory>::Config, MatmulLaunchError> {
        let config =
            Self::BatchMatmul::make_config(input, problem, cube_dim, cube_count, quantized);
        problem.check_config(&config)?;
        Self::BatchMatmul::check_config(&config)?;
        Ok(config)
    }

    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &<Self::BatchMatmul as MatmulConfigFactory>::Config,
    ) -> Result<(), MatmulAvailabilityError> {
        Self::BatchMatmul::check_availability::<R, MP>(client, config)
    }
}
