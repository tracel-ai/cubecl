use cubecl_core::prelude::*;

use cubecl_core::Runtime;

use crate::matmul::components::batch;
use crate::matmul::components::global;
use crate::matmul::components::stage;
use crate::matmul::components::stage::Matmul as StageMatmul;
use crate::matmul::components::tile::Matmul as TileMatmul;
use crate::matmul::components::MatmulProblem;
use crate::matmul::components::StageDim;

use super::dispatch::MatmulLaunchDispatch;

// pub(crate) type CmmaTmmConfig = tile::plane::Config;
pub(crate) type CmmaSmmConfig<T> = stage::row_accumulate::Config<T>;
pub(crate) type CmmaGmmConfig<T> = global::homogeneous::Config<CmmaSmmConfig<T>>;
pub(crate) type CmmaBmmConfig<T> = batch::one_to_one::Config<CmmaGmmConfig<T>>;

/// Configs that should not hinder correctness, but may impact performance
pub struct AdvancedConfig {
    pub tiling_order: stage::TilingOrderConfig,
}

impl Default for AdvancedConfig {
    fn default() -> Self {
        Self {
            tiling_order: stage::TilingOrderConfig::XMajor,
        }
    }
}

/// Make a config for the cmma batch kernel, given problem definition,
/// cube settings and advanced config
pub fn make_cmma_config<EG, D, R>(
    problem: &MatmulProblem<EG>,
    cube_dim: &CubeDim,
    cube_count: &CubeCount,
    advanced_config: &AdvancedConfig,
) -> CmmaBmmConfig<D::TileConfig>
where
    EG: Numeric,
    D: MatmulLaunchDispatch,
    R: Runtime,
{
    type TMM<D> = <D as MatmulLaunchDispatch>::TileMatmul;
    type SMM<D, EG> = stage::row_accumulate::Matmul<
        <D as MatmulLaunchDispatch>::ElementInput,
        EG,
        <D as MatmulLaunchDispatch>::ElementAccumulator,
        TMM<D>,
        <D as MatmulLaunchDispatch>::StageSize,
        CmmaSmmConfig<<D as MatmulLaunchDispatch>::TileConfig>,
    >;

    let (stage_m, stage_n, stage_k) = (SMM::<D, EG>::M, SMM::<D, EG>::N, SMM::<D, EG>::K);
    let (tile_m, tile_n, tile_k) = (TMM::<D>::M, TMM::<D>::N, TMM::<D>::K);
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

    let s = CmmaSmmConfig::new(
        D::tile_config(plane_dim, &problem),
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
    problem.check_config::<CmmaBmmConfig<D::TileConfig>>(&b);

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
