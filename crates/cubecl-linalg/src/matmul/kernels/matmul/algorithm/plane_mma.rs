use std::marker::PhantomData;

use cubecl_core::prelude::*;

use crate::matmul::components::batch::CubeCountDispatch;
use crate::matmul::components::stage::{self, S4x4x2, StageSize};
use crate::matmul::components::tile::plane::PlaneMma16x16x16;
use crate::matmul::components::tile::Matmul;
use crate::matmul::components::MatmulProblem;
use crate::matmul::components::{batch, global};

use super::base;

pub struct PlaneMma<EG> {
    pub _eg: PhantomData<EG>,
}

type Dispatch = batch::NaturalDispatch;
type Stage = S4x4x2;

impl<EG: Numeric> base::Algorithm<EG> for PlaneMma<EG> {
    const PLANE_DIM: u32 = 32;
    type EG = EG;
    type ES = f32;
    type EA = f32;

    type TileMatmul = PlaneMma16x16x16<Self::ES, Self::EA>;

    type StageMatmul =
        stage::multi_buffer::Matmul<Self::ES, Self::EG, Self::EA, Self::TileMatmul, Stage>;

    type GlobalMatmul = global::homogeneous::Matmul<
        Self::EG,
        Self::ES,
        Self::StageMatmul,
        global::homogeneous::CyclicLoading,
        global::homogeneous::CyclicLoading,
    >;

    type BatchMatmul = batch::one_to_one::Matmul<Self::EG, Self::ES, Self::GlobalMatmul, Dispatch>;

    fn cube_dim() -> CubeDim {
        CubeDim::new(Self::PLANE_DIM, Stage::NUM_M, 1)
    }

    fn cube_count(problem: &MatmulProblem) -> CubeCount {
        let m_stage = Stage::NUM_M * Self::TileMatmul::M;
        let n_stage = Stage::NUM_N * Self::TileMatmul::N;
        let cubes_for_m = (problem.m as u32 + m_stage - 1) / m_stage;
        let cubes_for_n = (problem.n as u32 + n_stage - 1) / n_stage;

        Dispatch::cube_count(cubes_for_m, cubes_for_n, problem.num_batches() as u32)
    }
}
