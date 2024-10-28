use cubecl_core::prelude::*;
use cubecl_core::Runtime;

use crate::matmul::cmma_matmul::batch::CmmaBatchMatmulConfig;
use crate::matmul::cmma_matmul::global::CmmaGlobalMatmulConfig;
use crate::matmul::cmma_matmul::global::LhsTensorLoader;
use crate::matmul::cmma_matmul::global::RhsTensorLoader;
use crate::matmul::cmma_matmul::global::TensorUnloader;
use crate::matmul::cmma_matmul::stage::CmmaStageMatmulConfig;
use crate::matmul::cmma_matmul::stage::LhsStageReader;
use crate::matmul::cmma_matmul::stage::RhsStageReader;
use crate::matmul::cmma_matmul::stage::TilingOrderConfig;
use crate::matmul::cmma_matmul::tile::CmmaTileMatmulConfig;
use crate::matmul::matmul_batch::BatchMatmul;
use crate::matmul::stage_dim::StageDim;
use crate::matmul::{
    matmul_global::GlobalMatmul, matmul_stage::StageMatmul, matmul_tile::TileMatmul,
    problem::MatmulProblem,
};

type T = CmmaTileMatmulConfig;
type S = CmmaStageMatmulConfig<T>;
type G = CmmaGlobalMatmulConfig<S>;
pub type CmmaConfig = CmmaBatchMatmulConfig<G>;

pub struct AdvancedConfig {
    pub tiling_order: TilingOrderConfig,
}

impl Default for AdvancedConfig {
    fn default() -> Self {
        Self {
            tiling_order: TilingOrderConfig::XMajor,
        }
    }
}

pub fn make_cmma_config<EG, ES, EA, TMM, SMM, GMM, BMM, R>(
    problem: &MatmulProblem,
    cube_dim: &CubeDim,
    cube_count: &CubeCount,
    advanced_config: &AdvancedConfig,
) -> CmmaConfig
where
    TMM: TileMatmul<ES, EA, T>,
    SMM: StageMatmul<ES, EG, LhsStageReader<ES, S>, RhsStageReader<ES, S>, S>,
    GMM: GlobalMatmul<
        EG,
        ES,
        LhsTensorLoader<EG, ES, G>,
        RhsTensorLoader<EG, ES, G>,
        TensorUnloader<EG, G>,
        G,
    >,
    BMM: BatchMatmul<EG, CmmaConfig>,
    EG: Numeric,
    ES: Numeric,
    EA: Numeric,
    R: Runtime,
{
    let (stage_m, stage_n, stage_k) = (SMM::M, SMM::N, SMM::K);
    let (tile_m, tile_n, tile_k) = (TMM::M, TMM::N, TMM::K);
    let (lhs_stage_dim, rhs_stage_dim, out_stage_dim) =
        create_stage_dim(stage_m, stage_n, stage_k, tile_m, tile_n, tile_k);

    let check_m_bounds = problem.m as u32 % stage_m != 0;
    let check_n_bounds = problem.n as u32 % stage_n != 0;

    let plane_dim = cube_dim.x;
    let num_planes = cube_dim.y;

    let (cube_count_x, cube_count_y, cube_count_z) = if let CubeCount::Static(x, y, z) = cube_count
    {
        (x, y, z)
    } else {
        panic!("Dynamic cube count unsupported")
    };

    let t = T::new(
        plane_dim,
        problem.lhs_layout,
        problem.rhs_layout,
        problem.lhs_line_size as u32,
        problem.rhs_line_size as u32,
        problem.out_line_size as u32,
    );
    let s = S::new(
        t,
        lhs_stage_dim,
        rhs_stage_dim,
        out_stage_dim,
        num_planes,
        advanced_config.tiling_order,
    );
    let g = G::new(
        s,
        problem.out_line_size as u32,
        check_m_bounds,
        check_n_bounds,
    );
    let b = CmmaConfig::new(g, *cube_count_x, *cube_count_y, *cube_count_z);
    problem.check_config::<CmmaConfig>(&b);

    b
}

pub fn create_stage_dim(
    stage_m: u32,
    stage_n: u32,
    stage_k: u32,
    tile_m: u32,
    tile_n: u32,
    tile_k: u32,
) -> (StageDim, StageDim, StageDim) {
    let lhs_stage_dim = StageDim {
        tile_size_x: tile_m,
        tile_size_y: tile_k,
        num_tiles_x: stage_m / tile_m,
        num_tiles_y: stage_k / tile_k,
    };

    let rhs_stage_dim = StageDim {
        tile_size_x: tile_k,
        tile_size_y: tile_n,
        num_tiles_x: stage_k / tile_k,
        num_tiles_y: stage_n / tile_n,
    };

    let out_stage_dim = StageDim {
        tile_size_x: tile_m,
        tile_size_y: tile_n,
        num_tiles_x: stage_m / tile_m,
        num_tiles_y: stage_n / tile_n,
    };

    (lhs_stage_dim, rhs_stage_dim, out_stage_dim)
}
