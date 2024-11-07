use cubecl_core::prelude::*;

use crate::matmul::components::batch;
use crate::matmul::components::global;
use crate::matmul::components::stage;
use crate::matmul::components::stage::Matmul as _;
use crate::matmul::components::tile::Matmul as _;
use crate::matmul::components::MatmulKernel;
use crate::matmul::components::MatmulProblem;
use crate::matmul::components::MatrixLayout;
use crate::matmul::components::StageDim;

use super::dispatch::MatmulLaunchDispatch;

pub(crate) type CmmaSmmConfig<T> = stage::row_accumulate::Config<T>;
pub(crate) type CmmaGmmConfig<T> = global::homogeneous::Config<CmmaSmmConfig<T>>;
pub(crate) type CmmaBmmConfig<T> = batch::one_to_one::Config<CmmaGmmConfig<T>>;

/// Configs that may impact performance
///
pub struct AdvancedConfig {
    /// Order in which tiles should be in shared memory
    pub tiling_order: stage::TilingOrderConfig,
    /// Ensure the inputs to tile matmul are in specified layout
    ///
    /// # Notes
    ///
    /// First item is for LHS, second item is for RHS
    /// If None, the layout will be the same as in global memory
    /// If enforced layout is different from global memory,
    /// transpose will be done at loading from global memory to stage,
    /// and stage will not be vectorized.
    pub enforced_tile_layout: (Option<MatrixLayout>, Option<MatrixLayout>),
}

impl Default for AdvancedConfig {
    fn default() -> Self {
        Self {
            tiling_order: stage::TilingOrderConfig::XMajor,
            enforced_tile_layout: (None, None),
        }
    }
}

/// Make a config for the cmma batch kernel, given problem definition,
/// cube settings and advanced config
pub fn make_cmma_config<EG, D>(
    problem: &MatmulProblem<EG>,
    cube_dim: &CubeDim,
    cube_count: &CubeCount,
    advanced_config: &AdvancedConfig,
) -> CmmaBmmConfig<<D::TileMatmul as MatmulKernel<D::ElementInput, D::ElementAccumulator>>::Config>
where
    EG: Numeric,
    D: MatmulLaunchDispatch,
{
    type Tmm<D> = <D as MatmulLaunchDispatch>::TileMatmul;
    type Smm<D, EG> = stage::row_accumulate::Matmul<
        <D as MatmulLaunchDispatch>::ElementInput,
        EG,
        <D as MatmulLaunchDispatch>::ElementAccumulator,
        Tmm<D>,
        <D as MatmulLaunchDispatch>::StageSize,
    >;

    let (stage_m, stage_n, stage_k) = (Smm::<D, EG>::M, Smm::<D, EG>::N, Smm::<D, EG>::K);
    let (tile_m, tile_n, tile_k) = (Tmm::<D>::M, Tmm::<D>::N, Tmm::<D>::K);
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

    let (lhs_tile_layout, lhs_tile_line_size) = match advanced_config.enforced_tile_layout.0 {
        Some(enforced_layout) if enforced_layout != problem.lhs_layout => (enforced_layout, 1),
        _ => (problem.lhs_layout, problem.lhs_line_size),
    };

    let (rhs_tile_layout, rhs_tile_line_size) = match advanced_config.enforced_tile_layout.1 {
        Some(enforced_layout) if enforced_layout != problem.rhs_layout => (enforced_layout, 1),
        _ => (problem.rhs_layout, problem.rhs_line_size),
    };

    let s = CmmaSmmConfig::new(
        D::tile_config(
            plane_dim,
            lhs_tile_layout,
            rhs_tile_layout,
            lhs_tile_line_size as u32,
            rhs_tile_line_size as u32,
            problem.out_line_size as u32,
        ),
        lhs_stage_dim,
        rhs_stage_dim,
        out_stage_dim,
        num_planes,
        advanced_config.tiling_order,
    );
    let g = CmmaGmmConfig::new(
        s,
        check_m_bounds,
        check_n_bounds,
        problem.lhs_layout,
        problem.rhs_layout,
        problem.lhs_line_size as u32,
        problem.rhs_line_size as u32,
        problem.out_line_size as u32,
    );
    let b = CmmaBmmConfig::new(g, *cube_count_x, *cube_count_y, *cube_count_z);
    problem.check_config::<CmmaBmmConfig<
        <D::TileMatmul as MatmulKernel<D::ElementInput, D::ElementAccumulator>>::Config,
    >>(&b);

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
