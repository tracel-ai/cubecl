use cubecl_core::prelude::*;

use crate::matmul::matmul_modular::cmma_matmul::stage::{S4x4x2, StageSize};
use crate::matmul::matmul_modular::cmma_matmul::tile::{
    CmmaInstruction16_16_16, CmmaTileMatmulConfig, PlaneMma16x16x16,
};
use crate::matmul::matmul_modular::matmul_tile::TileMatmul;
use crate::matmul::matmul_modular::problem::MatmulProblem;

/// Launch informations for a matmul
pub trait MatmulLaunchDispatch {
    const PLANE_DIM: u32;
    type StageSize: StageSize;
    type ElementInput: Numeric;
    type ElementAccumulator: Numeric;

    type TileMatmul: TileMatmul<Self::ElementInput, Self::ElementAccumulator, CmmaTileMatmulConfig>;

    fn cube_dim() -> CubeDim;
    fn cube_count<EG: Numeric>(problem: &MatmulProblem<EG>) -> CubeCount;
}

pub struct PlaneMmaLaunchDispatch {}

impl MatmulLaunchDispatch for PlaneMmaLaunchDispatch {
    const PLANE_DIM: u32 = 32;
    type StageSize = S4x4x2;
    type ElementInput = f32;
    type ElementAccumulator = f32;

    type TileMatmul =
        PlaneMma16x16x16<Self::ElementInput, Self::ElementAccumulator, CmmaTileMatmulConfig>;

    fn cube_dim() -> CubeDim {
        CubeDim::new(Self::PLANE_DIM, Self::StageSize::NUM_M, 1)
    }

    fn cube_count<EG: Numeric>(problem: &MatmulProblem<EG>) -> CubeCount {
        let m_stage = Self::StageSize::NUM_M * Self::TileMatmul::M;
        let n_stage = Self::StageSize::NUM_N * Self::TileMatmul::N;
        let cubes_needed_m = (problem.m as u32 + m_stage - 1) / m_stage;
        let cubes_needed_n = (problem.n as u32 + n_stage - 1) / n_stage;

        CubeCount::Static(cubes_needed_m, cubes_needed_n, problem.num_batches() as u32)
    }
}

pub struct CmmaLaunchDispatch {}

impl MatmulLaunchDispatch for CmmaLaunchDispatch {
    const PLANE_DIM: u32 = 32;
    type StageSize = S4x4x2;
    type ElementInput = half::f16;
    type ElementAccumulator = f32;

    type TileMatmul =
        CmmaInstruction16_16_16<Self::ElementInput, Self::ElementAccumulator, CmmaTileMatmulConfig>;

    fn cube_dim() -> CubeDim {
        CubeDim::new(Self::PLANE_DIM, Self::StageSize::NUM_M, 1)
    }

    fn cube_count<EG: Numeric>(problem: &MatmulProblem<EG>) -> CubeCount {
        let m_stage = Self::StageSize::NUM_M * Self::TileMatmul::M;
        let n_stage = Self::StageSize::NUM_N * Self::TileMatmul::N;
        let cubes_needed_m = (problem.m as u32 + m_stage - 1) / m_stage;
        let cubes_needed_n = (problem.n as u32 + n_stage - 1) / n_stage;

        CubeCount::Static(cubes_needed_m, cubes_needed_n, problem.num_batches() as u32)
    }
}
