use std::marker::PhantomData;

use cubecl_core::prelude::*;

use crate::matmul::components::batch::CubeCountDispatch;
use crate::matmul::components::stage::{self, StageSize};
use crate::matmul::components::tile;
use crate::matmul::components::MatmulProblem;
use crate::matmul::components::{batch, global};

use super::base;

type Dispatch = batch::SwizzleTransposedDispatch<2>;

pub struct StandardAlgorithm<
    EG: Numeric,
    ES: Numeric,
    EA: Numeric,
    Stage: StageSize,
    TMM: tile::Matmul<ES, EA>,
> {
    pub _eg: PhantomData<EG>,
    pub _es: PhantomData<ES>,
    pub _ea: PhantomData<EA>,
    pub _stage: PhantomData<Stage>,
    pub _tmm: PhantomData<TMM>,
}

impl<EG: Numeric, ES: Numeric, EA: Numeric, Stage: StageSize, TMM: tile::Matmul<ES, EA>>
    base::Algorithm<EG> for StandardAlgorithm<EG, ES, EA, Stage, TMM>
{
    const PLANE_DIM: u32 = 32;

    type EG = EG;
    type ES = ES;
    type EA = EA;

    type TileMatmul = TMM;

    type StageMatmul =
        stage::single_buffer::Matmul<Self::ES, Self::EG, Self::EA, Self::TileMatmul, Stage>;

    type GlobalMatmul =
        global::buffered::specialized::Matmul<Self::EG, Self::ES, Self::EA, Self::StageMatmul>;

    type BatchMatmul = batch::one_to_one::Matmul<Self::EG, Self::ES, Self::GlobalMatmul, Dispatch>;

    fn cube_dim() -> CubeDim {
        CubeDim::new(Self::PLANE_DIM, Stage::NUM_M + 4, 1)
    }

    fn cube_count(problem: &MatmulProblem) -> CubeCount {
        let m_stage = Stage::NUM_M * Self::TileMatmul::M;
        let n_stage = Stage::NUM_N * Self::TileMatmul::N;
        let cubes_for_m = (problem.m as u32 + m_stage - 1) / m_stage;
        let cubes_for_n = (problem.n as u32 + n_stage - 1) / n_stage;

        Dispatch::cube_count(cubes_for_m, cubes_for_n, problem.num_batches() as u32)
    }

    fn advanced_config() -> crate::matmul::kernels::matmul::AdvancedConfig {
        crate::matmul::kernels::matmul::AdvancedConfig {
            lhs_tiling_order: stage::TilingOrderConfig::ColMajor,
            rhs_tiling_order: stage::TilingOrderConfig::RowMajor,
            enforced_tile_layout: (None, None),
        }
    }
}
