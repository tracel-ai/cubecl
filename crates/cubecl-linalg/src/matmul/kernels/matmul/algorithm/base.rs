use cubecl_core::prelude::*;

use crate::matmul::components::stage::{self};
use crate::matmul::components::{batch, global, tile, MatmulSpec};
use crate::matmul::components::{MatmulKernel, MatmulProblem};
use crate::matmul::kernels::matmul::AdvancedConfig;
use crate::matmul::kernels::{MatmulAvailabilityError, MatmulInvalidProblem};

type LhsStageReader<MS, GMM> = <<GMM as global::Matmul<MS>>::LhsLoader as global::Loader<
    EG<MS>,
    ES<MS>,
    <GMM as MatmulKernel>::Config,
>>::StageReader;
type RhsStageReader<MS, GMM> = <<GMM as global::Matmul<MS>>::RhsLoader as global::Loader<
    EG<MS>,
    ES<MS>,
    <GMM as MatmulKernel>::Config,
>>::StageReader;

type EG<MS> = <MS as MatmulSpec>::EG;
type ES<MS> = <MS as MatmulSpec>::ES;

/// Specifications for a matmul algorithm
pub trait Algorithm<MS: MatmulSpec> {
    const PLANE_DIM: u32;

    type TileMatmul: tile::Matmul<MS::ES, MS::EA> + MatmulKernel;

    type StageMatmul: stage::Matmul<
            MS::ES,
            MS::EG,
            MS::EA,
            LhsReader = LhsStageReader<MS, Self::GlobalMatmul>,
            RhsReader = RhsStageReader<MS, Self::GlobalMatmul>,
        > + MatmulKernel;

    type GlobalMatmul: global::Matmul<MS>;

    type BatchMatmul: batch::Matmul<MS> + MatmulKernel;

    fn cube_dim() -> CubeDim;
    fn cube_count(problem: &MatmulProblem) -> CubeCount;
    #[allow(clippy::type_complexity)]
    fn make_config(
        problem: &MatmulProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        advanced_config: &AdvancedConfig,
    ) -> Result<<Self::BatchMatmul as MatmulKernel>::Config, MatmulInvalidProblem> {
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
