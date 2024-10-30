use cubecl_core::prelude::*;

use cubecl_core::Runtime;

use crate::matmul::components::batch::BatchMatmul;
use crate::matmul::components::batch::OneToOneBatchMatmulConfig;
use crate::matmul::components::cmma_matmul::global::{
    HomogeneousGlobalMatmulConfig, LhsTensorLoader, RhsTensorLoader, TensorUnloader,
};
use crate::matmul::components::global::GlobalMatmul;
use crate::matmul::components::problem::MatmulProblem;
use crate::matmul::components::stage::LhsStageReader;
use crate::matmul::components::stage::RowAccumulateStageMatmulConfig;
use crate::matmul::components::stage::RhsStageReader;
use crate::matmul::components::stage::StageMatmul;
use crate::matmul::components::stage::TilingOrderConfig;
use crate::matmul::components::stage_dim::StageDim;
use crate::matmul::components::tile::TileConfig;
use crate::matmul::components::tile::TileMatmul;

pub(crate) type CmmaTmmConfig = TileConfig;
pub(crate) type CmmaSmmConfig = RowAccumulateStageMatmulConfig<CmmaTmmConfig>;
pub(crate) type CmmaGmmConfig = HomogeneousGlobalMatmulConfig<CmmaSmmConfig>;
pub(crate) type CmmaBmmConfig = OneToOneBatchMatmulConfig<CmmaGmmConfig>;

/// Configs that should not hinder correctness, but may impact performance
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

/// Make a config for the cmma batch kernel, given problem definition,
/// cube settings and advanced config
pub fn make_cmma_config<EG, ES, EA, TMM, SMM, GMM, BMM, R>(
    problem: &MatmulProblem<EG>,
    cube_dim: &CubeDim,
    cube_count: &CubeCount,
    advanced_config: &AdvancedConfig,
) -> CmmaBmmConfig
where
    TMM: TileMatmul<ES, EA, CmmaTmmConfig>,
    SMM: StageMatmul<
        ES,
        EG,
        LhsStageReader<ES, CmmaSmmConfig>,
        RhsStageReader<ES, CmmaSmmConfig>,
        CmmaSmmConfig,
    >,
    GMM: GlobalMatmul<
        EG,
        ES,
        LhsTensorLoader<EG, ES, CmmaGmmConfig>,
        RhsTensorLoader<EG, ES, CmmaGmmConfig>,
        TensorUnloader<EG, CmmaGmmConfig>,
        CmmaGmmConfig,
    >,
    BMM: BatchMatmul<EG, CmmaBmmConfig>,
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

    let t = CmmaTmmConfig::new(
        plane_dim,
        problem.lhs_layout,
        problem.rhs_layout,
        problem.lhs_line_size as u32,
        problem.rhs_line_size as u32,
        problem.out_line_size as u32,
    );
    let s = CmmaSmmConfig::new(
        t,
        lhs_stage_dim,
        rhs_stage_dim,
        out_stage_dim,
        num_planes,
        advanced_config.tiling_order,
    );
    let g = CmmaGmmConfig::new(
        s,
        problem.out_line_size as u32,
        check_m_bounds,
        check_n_bounds,
    );
    let b = CmmaBmmConfig::new(g, *cube_count_x, *cube_count_y, *cube_count_z);
    problem.check_config::<CmmaBmmConfig>(&b);

    b
}

fn create_stage_dim(
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
