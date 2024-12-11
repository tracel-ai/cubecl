use std::marker::PhantomData;

use cubecl_core::prelude::*;

use crate::matmul::components::batch::CubeCountDispatch;
use crate::matmul::components::global::full_load::CyclicLoading;
use crate::matmul::components::stage::{self, StageSize};
use crate::matmul::components::MatmulProblem;
use crate::matmul::components::{batch, global};
use crate::matmul::components::{tile, MatmulSpec};

use super::base;

type Dispatch = batch::SwizzleTransposedDispatch<2>;

pub struct StandardAlgorithm<MS: MatmulSpec, Stage: StageSize, TMM> {
    pub _ms: PhantomData<MS>,
    pub _stage: PhantomData<Stage>,
    pub _tmm: PhantomData<TMM>,
}

impl<MS: MatmulSpec, Stage: StageSize, TMM: tile::Matmul<MS::ES, MS::EA>> base::Algorithm<MS>
    for StandardAlgorithm<MS, Stage, TMM>
{
    const PLANE_DIM: u32 = 32;

    type TileMatmul = TMM;
    type StageMatmul = stage::multi_buffer::Matmul<MS::ES, MS::EG, MS::EA, Self::TileMatmul, Stage>;
    type GlobalMatmul =
        global::full_load::Matmul<MS, Self::StageMatmul, CyclicLoading, CyclicLoading>;

    type BatchMatmul = batch::one_to_one::Matmul<MS, Self::GlobalMatmul, Dispatch>;

    fn cube_dim() -> CubeDim {
        CubeDim::new(<Self as base::Algorithm<MS>>::PLANE_DIM, Stage::NUM_M, 1)
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
