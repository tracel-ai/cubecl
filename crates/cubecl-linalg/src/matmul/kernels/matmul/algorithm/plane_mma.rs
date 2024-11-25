use std::marker::PhantomData;

use cubecl_core::prelude::*;

use crate::matmul::components::stage::{self, S4x4x2, StageSize};
use crate::matmul::components::tile::plane::PlaneMma16x16x16;
use crate::matmul::components::tile::Matmul;
use crate::matmul::components::MatmulProblem;
use crate::matmul::components::{batch, global};

use super::base;

pub struct PlaneMma<EG> {
    pub _eg: PhantomData<EG>,
}

impl<EG: Numeric> base::Algorithm<EG> for PlaneMma<EG> {
    const PLANE_DIM: u32 = 32;
    type EG = EG;
    type ES = f32;
    type EA = f32;

    type TileMatmul = PlaneMma16x16x16<Self::ES, Self::EA>;

    type StageMatmul =
        stage::multi_buffer::Matmul<Self::ES, Self::EG, Self::EA, Self::TileMatmul, S4x4x2>;

    type GlobalMatmul = global::homogeneous::Matmul<Self::EG, Self::ES, Self::StageMatmul>;

    type BatchMatmul =
        batch::one_to_one::Matmul<Self::EG, Self::ES, Self::GlobalMatmul, batch::NaturalDispatch>;

    fn cube_dim() -> CubeDim {
        CubeDim::new(Self::PLANE_DIM, S4x4x2::NUM_M, 1)
    }

    fn cube_count(problem: &MatmulProblem) -> CubeCount {
        let m_stage = S4x4x2::NUM_M * Self::TileMatmul::M;
        let n_stage = S4x4x2::NUM_N * Self::TileMatmul::K;
        let cubes_needed_m = (problem.m as u32 + m_stage - 1) / m_stage;
        let cubes_needed_n = (problem.n as u32 + n_stage - 1) / n_stage;

        CubeCount::Static(cubes_needed_m, cubes_needed_n, problem.num_batches() as u32)
    }
}
