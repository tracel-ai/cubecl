use cubecl_core::prelude::*;

use crate::matmul::components::global::args::GmmArgs;
use crate::matmul::components::stage::{self};
use crate::matmul::components::{batch, global, tile};
use crate::matmul::components::{MatmulKernel, MatmulProblem};
use crate::matmul::kernels::matmul::AdvancedConfig;
use crate::matmul::kernels::{MatmulAvailabilityError, MatmulInvalidProblem};

type LhsStageReader<GA, GMM, EG, ES> =
    <<GMM as global::Matmul<GA, EG, ES>>::LhsLoader as global::Loader<
        EG,
        ES,
        <GMM as MatmulKernel<EG, EG>>::Config,
    >>::StageReader;
type RhsStageReader<GA, GMM, EG, ES> =
    <<GMM as global::Matmul<GA, EG, ES>>::RhsLoader as global::Loader<
        EG,
        ES,
        <GMM as MatmulKernel<EG, EG>>::Config,
    >>::StageReader;

/// Specifications for a matmul algorithm
pub trait Algorithm<GA: GmmArgs<EG>, EG: Numeric> {
    const PLANE_DIM: u32;

    type ES: Numeric;
    type EA: Numeric;

    type TileMatmul: tile::Matmul<Self::ES, Self::EA> + MatmulKernel<Self::ES, Self::EA>;

    type StageMatmul: stage::Matmul<
            Self::ES,
            EG,
            Self::EA,
            LhsReader = LhsStageReader<GA, Self::GlobalMatmul, EG, Self::ES>,
            RhsReader = RhsStageReader<GA, Self::GlobalMatmul, EG, Self::ES>,
        > + MatmulKernel<Self::ES, EG>;

    type GlobalMatmul: global::Matmul<GA, EG, Self::ES>;

    type BatchMatmul: batch::Matmul<GA, EG> + MatmulKernel<EG, EG>;

    fn cube_dim() -> CubeDim;
    fn cube_count(problem: &MatmulProblem) -> CubeCount;
    #[allow(clippy::type_complexity)]
    fn make_config(
        problem: &MatmulProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        advanced_config: &AdvancedConfig,
    ) -> Result<<Self::BatchMatmul as MatmulKernel<EG, EG>>::Config, MatmulInvalidProblem> {
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
