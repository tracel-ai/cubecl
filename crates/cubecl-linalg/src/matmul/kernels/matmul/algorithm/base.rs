use crate::matmul::components::stage::StageVectorization;
use crate::matmul::components::{
    CompleteStageTiling, MatmulConfigFactory, MatmulPrecision, MatmulProblem, MatmulSelection,
    batch, global, stage, tile,
};
use crate::matmul::kernels::{MatmulAvailabilityError, MatmulLaunchError};
use cubecl_core::prelude::*;

type StageInput = (
    CompleteStageTiling,
    stage::StageBuffering,
    StageVectorization,
);

/// Specifications for a matmul algorithm
pub trait Algorithm {
    type TileMatmul: tile::TileMatmulFamily;
    type StageMatmul: stage::StageMatmulFamily<Input = StageInput>;
    type GlobalMatmul: global::GlobalMatmulFamily;
    type BatchMatmul: batch::BatchMatmulFamily<Input = StageInput>;

    fn cube_dim(selection: &MatmulSelection) -> CubeDim;
    fn cube_count(selection: &MatmulSelection, problem: &MatmulProblem) -> CubeCount;

    #[allow(clippy::type_complexity, clippy::result_large_err)]
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

    #[allow(clippy::result_large_err)]
    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &<Self::BatchMatmul as MatmulConfigFactory>::Config,
    ) -> Result<(), MatmulAvailabilityError> {
        Self::BatchMatmul::check_availability::<R, MP>(client, config)
    }
}
