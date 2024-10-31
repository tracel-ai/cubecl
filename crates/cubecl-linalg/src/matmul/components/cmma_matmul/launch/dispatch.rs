use cubecl_core::prelude::*;

use crate::matmul::components::problem::MatmulProblem;
use crate::matmul::components::stage::{S4x4x2, StageSize};
use crate::matmul::components::tile;
use crate::matmul::components::tile::accelerated::Accelerated16x16x16;
use crate::matmul::components::tile::plane::PlaneMma16x16x16;
use crate::matmul::components::tile::Matmul;

/// Launch informations for a matmul
pub trait MatmulLaunchDispatch {
    const PLANE_DIM: u32;
    type StageSize: StageSize;
    type ElementInput: Numeric;
    type ElementAccumulator: Numeric;

    type TileConfig: tile::Config;
    type TileMatmul: tile::Matmul<Self::ElementInput, Self::ElementAccumulator, Self::TileConfig>;

    fn cube_dim() -> CubeDim;
    fn cube_count<EG: Numeric>(problem: &MatmulProblem<EG>) -> CubeCount;
    fn tile_config<EG: Numeric>(plane_dim: u32, problem: &MatmulProblem<EG>) -> Self::TileConfig;
}

pub struct PlaneMmaLaunchDispatch {}

impl MatmulLaunchDispatch for PlaneMmaLaunchDispatch {
    const PLANE_DIM: u32 = 32;
    type StageSize = S4x4x2;
    type ElementInput = f32;
    type ElementAccumulator = f32;

    type TileConfig = tile::plane::Config;
    type TileMatmul =
        PlaneMma16x16x16<Self::ElementInput, Self::ElementAccumulator, Self::TileConfig>;

    fn cube_dim() -> CubeDim {
        CubeDim::new(Self::PLANE_DIM, Self::StageSize::NUM_M, 1)
    }

    fn cube_count<EG: Numeric>(problem: &MatmulProblem<EG>) -> CubeCount {
        let m_stage = Self::StageSize::NUM_M * Self::TileMatmul::M;
        let n_stage = Self::StageSize::NUM_N * Self::TileMatmul::K;
        let cubes_needed_m = (problem.m as u32 + m_stage - 1) / m_stage;
        let cubes_needed_n = (problem.n as u32 + n_stage - 1) / n_stage;

        CubeCount::Static(cubes_needed_m, cubes_needed_n, problem.num_batches() as u32)
    }

    fn tile_config<EG: Numeric>(plane_dim: u32, problem: &MatmulProblem<EG>) -> Self::TileConfig {
        Self::TileConfig::new(
            plane_dim,
            problem.lhs_layout,
            problem.rhs_layout,
            problem.lhs_line_size as u32,
            problem.rhs_line_size as u32,
            problem.out_line_size as u32,
        )
    }
}

pub struct CmmaLaunchDispatch {}

impl MatmulLaunchDispatch for CmmaLaunchDispatch {
    const PLANE_DIM: u32 = 32;
    type StageSize = S4x4x2;
    type ElementInput = half::f16;
    type ElementAccumulator = f32;

    type TileConfig = tile::accelerated::Config;
    type TileMatmul =
        Accelerated16x16x16<Self::ElementInput, Self::ElementAccumulator, Self::TileConfig>;

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

    fn tile_config<EG: Numeric>(plane_dim: u32, problem: &MatmulProblem<EG>) -> Self::TileConfig {
        Self::TileConfig::new(
            plane_dim,
            problem.lhs_layout,
            problem.rhs_layout,
            problem.lhs_line_size as u32,
            problem.rhs_line_size as u32,
            problem.out_line_size as u32,
        )
    }
}
