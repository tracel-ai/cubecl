use std::marker::PhantomData;

use cubecl_core::{prelude::*, Feature};

use crate::matmul::components::stage::{self, S4x4x2, StageSize};
use crate::matmul::components::tile::plane::PlaneMma16x16x16;
use crate::matmul::components::tile::Matmul;
use crate::matmul::components::{batch, global};
use crate::matmul::components::{MatmulKernel, MatmulProblem};
use crate::matmul::kernels::cmma_matmul::AdvancedConfig;

use super::base;

pub struct Algorithm<EG> {
    _eg: PhantomData<EG>,
}

impl<EG: Numeric> base::Algorithm<EG> for Algorithm<EG> {
    const PLANE_DIM: u32 = 32;
    type EG = EG;
    type ES = f32;
    type EA = f32;

    type TileMatmul = PlaneMma16x16x16<Self::ES, Self::EA>;

    type StageSize = S4x4x2;
    type StageMatmul = stage::row_accumulate::Matmul<
        Self::ES,
        Self::EG,
        Self::EA,
        Self::TileMatmul,
        Self::StageSize,
    >;

    type LhsLoader = global::tensor_view::LhsLoader<Self::EG, Self::ES>;
    type RhsLoader = global::tensor_view::RhsLoader<Self::EG, Self::ES>;
    type Unloader = global::tensor_view::Unloader<Self::EG>;
    type GlobalMatmul = global::homogeneous::Matmul<Self::EG, Self::ES, Self::StageMatmul>;

    type BatchMatmul = batch::one_to_one::Matmul<Self::EG, Self::ES, Self::GlobalMatmul>;

    fn cube_dim() -> CubeDim {
        CubeDim::new(Self::PLANE_DIM, Self::StageSize::NUM_M, 1)
    }

    fn cube_count(problem: &MatmulProblem<EG>) -> CubeCount {
        let m_stage = Self::StageSize::NUM_M * Self::TileMatmul::M;
        let n_stage = Self::StageSize::NUM_N * Self::TileMatmul::K;
        let cubes_needed_m = (problem.m as u32 + m_stage - 1) / m_stage;
        let cubes_needed_n = (problem.n as u32 + n_stage - 1) / n_stage;

        CubeCount::Static(cubes_needed_m, cubes_needed_n, problem.num_batches() as u32)
    }

    fn make_config(
        problem: &MatmulProblem<Self::EG>,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        advanced_config: &AdvancedConfig,
    ) -> <Self::BatchMatmul as MatmulKernel<Self::EG, Self::EG>>::Config {
        todo!()
    }

    fn check_availability<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> Result<(), ()> {
        if !client.properties().feature_enabled(Feature::Subcube) {
            return Err(());
        }

        if !(client
            .properties()
            .feature_enabled(Feature::Type(Self::EG::as_elem()))
            && client
                .properties()
                .feature_enabled(Feature::Type(Self::ES::as_elem()))
            && client
                .properties()
                .feature_enabled(Feature::Type(Self::EA::as_elem())))
        {
            return Err(());
        }

        Ok(())
    }
}
