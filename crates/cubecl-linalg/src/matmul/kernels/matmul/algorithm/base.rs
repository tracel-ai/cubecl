use cubecl_core::prelude::*;

use crate::matmul::components::stage::{self};
use crate::matmul::components::{batch, global, tile, MatmulSpec};
use crate::matmul::components::{MatmulConfigFactory, MatmulProblem};
use crate::matmul::kernels::matmul::AdvancedConfig;
use crate::matmul::kernels::{MatmulAvailabilityError, MatmulInvalidProblem};

type LhsStageReader<MS, GMM> = <<GMM as global::GlobalMatmul<MS>>::LhsLoader as global::Loader<
    EG<MS>,
    ES<MS>,
    <GMM as MatmulConfigFactory>::Config,
>>::StageReader;
type RhsStageReader<MS, GMM> = <<GMM as global::GlobalMatmul<MS>>::RhsLoader as global::Loader<
    EG<MS>,
    ES<MS>,
    <GMM as MatmulConfigFactory>::Config,
>>::StageReader;

type EG<MS> = <MS as MatmulSpec>::EG;
type ES<MS> = <MS as MatmulSpec>::ES;

/// Specifications for a matmul algorithm
pub trait Algorithm {
    type TileMatmul: tile::TileMatmulFamily;
    type StageMatmul: stage::MatmulFamily;
    type GlobalMatmul: global::GlobalMatmulFamily;
    type BatchMatmul: batch::BatchMatmulFamily;

    fn cube_dim() -> CubeDim;
    fn cube_count(problem: &MatmulProblem) -> CubeCount;
    #[allow(clippy::type_complexity)]
    fn make_config(
        problem: &MatmulProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        advanced_config: &AdvancedConfig,
    ) -> Result<<Self::BatchMatmul as MatmulConfigFactory>::Config, MatmulInvalidProblem> {
        let config = Self::BatchMatmul::make_config(problem, cube_dim, cube_count, advanced_config);
        problem.check_config(&config)?;
        Ok(config)
    }

    fn check_availability<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> Result<(), MatmulAvailabilityError> {
        Self::BatchMatmul::check_availability::<R>(client)
    }

    fn advanced_config() -> AdvancedConfig {
        AdvancedConfig::default()
    }
}
