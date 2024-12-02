use cubecl_core::prelude::*;

use crate::matmul::components::stage::{self};
use crate::matmul::components::{batch, global, tile};
use crate::matmul::components::{MatmulKernel, MatmulProblem};
use crate::matmul::kernels::matmul::AdvancedConfig;
use crate::matmul::kernels::{MatmulAvailabilityError, MatmulInvalidProblem};

type LhsStageReader<GMM, EG, ES> =
    <<GMM as global::Matmul<EG, ES>>::LhsLoader as global::Loader<EG, ES>>::StageReader;
type RhsStageReader<GMM, EG, ES> =
    <<GMM as global::Matmul<EG, ES>>::RhsLoader as global::Loader<EG, ES>>::StageReader;

/// Specifications for a matmul algorithm
pub trait Algorithm<EG: Numeric> {
    const PLANE_DIM: u32;

    type EG: Numeric;
    type ES: Numeric;
    type EA: Numeric;

    type TileMatmul: tile::Matmul<Self::ES, Self::EA> + MatmulKernel<Self::ES, Self::EA>;

    type StageMatmul: stage::Matmul<
            Self::ES,
            Self::EG,
            Self::EA,
            LhsReader = LhsStageReader<Self::GlobalMatmul, Self::EG, Self::ES>,
            RhsReader = RhsStageReader<Self::GlobalMatmul, Self::EG, Self::ES>,
        > + MatmulKernel<Self::ES, Self::EG>;

    type GlobalMatmul: global::Matmul<Self::EG, Self::ES> + MatmulKernel<Self::EG, Self::EG>;

    type BatchMatmul: batch::Matmul<Self::EG> + MatmulKernel<Self::EG, Self::EG>;

    fn cube_dim() -> CubeDim;
    fn cube_count(problem: &MatmulProblem) -> CubeCount;

    #[allow(clippy::type_complexity)]
    fn make_config(
        problem: &MatmulProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        advanced_config: &AdvancedConfig,
    ) -> Result<<Self::BatchMatmul as MatmulKernel<Self::EG, Self::EG>>::Config, MatmulInvalidProblem>
    {
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
