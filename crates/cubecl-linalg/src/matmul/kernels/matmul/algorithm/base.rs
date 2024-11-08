use cubecl_core::prelude::*;

use crate::matmul::components::stage::{self, StageSize};
use crate::matmul::components::{batch, global, tile};
use crate::matmul::components::{MatmulKernel, MatmulProblem};
use crate::matmul::kernels::matmul::AdvancedConfig;

/// Specifications for a matmul algorithm
pub trait Algorithm<EG: Numeric> {
    const PLANE_DIM: u32;

    type EG: Numeric;
    type ES: Numeric;
    type EA: Numeric;

    type TileMatmul: tile::Matmul<Self::ES, Self::EA> + MatmulKernel<Self::ES, Self::EA>;

    type StageSize: StageSize;
    type StageMatmul: stage::Matmul<
            Self::ES,
            Self::EG,
            <<Self::GlobalMatmul as global::Matmul<Self::EG, Self::ES>>::Lhs as global::Loader<
                Self::EG,
                Self::ES,
            >>::StageReader,
            <<Self::GlobalMatmul as global::Matmul<Self::EG, Self::ES>>::Rhs as global::Loader<
                Self::EG,
                Self::ES,
            >>::StageReader,
        > + MatmulKernel<Self::ES, Self::EG>;

    type GlobalMatmul: global::Matmul<Self::EG, Self::ES>;

    type BatchMatmul: batch::Matmul<Self::EG> + MatmulKernel<Self::EG, Self::EG>;

    fn cube_dim() -> CubeDim;
    fn cube_count(problem: &MatmulProblem) -> CubeCount;

    fn make_config(
        problem: &MatmulProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        advanced_config: &AdvancedConfig,
    ) -> <Self::BatchMatmul as MatmulKernel<Self::EG, Self::EG>>::Config {
        let config = Self::BatchMatmul::make_config(problem, cube_dim, cube_count, advanced_config);
        problem.check_config(&config);
        config
    }

    fn check_availability<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> Result<(), &str> {
        Self::BatchMatmul::check_availability::<R>(client)
    }
}
