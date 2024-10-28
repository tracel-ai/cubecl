use super::cmma_matmul::batch::{CmmaBatchMatmul, CmmaBatchMatmulConfig};
use super::cmma_matmul::global::{
    CmmaGlobalMatmul, CmmaGlobalMatmulConfig, LhsTensorLoader, RhsTensorLoader, TensorUnloader,
};
use super::cmma_matmul::stage::{
    CmmaStageMatmul, CmmaStageMatmulConfig, LhsStageReader, RhsStageReader, S4x4x2,
    TilingOrderConfig,
};
use super::cmma_matmul::tile::{
    check_cmma_availability, CmmaInstruction16_16_16, CmmaTileMatmulConfig, PlaneMma16x16x16,
};
use super::matmul_batch::BatchMatmul;
use super::matmul_global::GlobalMatmul;
use super::matmul_stage::StageMatmul;
use super::matmul_tile::TileMatmul;
use super::problem::MatmulProblem;
use super::stage_dim::StageDim;
use super::MatmulLaunch;
use cubecl_core::prelude::*;

use cubecl_core::{
    client::ComputeClient,
    frontend::{Float, TensorArg, TensorHandleRef},
    tensor_line_size, Runtime,
};

use crate::tensor::{into_contiguous, matrix_layout, MatrixLayout, TensorHandle};

pub fn matmul_cmma<R: Runtime, F: Float>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandle<R, F>,
    rhs: TensorHandle<R, F>,
    out: TensorHandle<R, F>,
    use_cmma_if_possible: bool,
) -> TensorHandle<R, F> {
    matmul_cmma_ref::<R, F>(
        client,
        lhs.as_ref(),
        rhs.as_ref(),
        out.as_ref(),
        use_cmma_if_possible,
    );
    out
}

pub fn matmul_cmma_ref<R: Runtime, F: Float>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandleRef<'_, R>,
    rhs: TensorHandleRef<'_, R>,
    out: TensorHandleRef<'_, R>,
    use_cmma_if_possible: bool,
) {
    let check_layout = |tensor: &TensorHandleRef<'_, R>| match matrix_layout(tensor.strides) {
        MatrixLayout::Contiguous => (false, false),
        MatrixLayout::MildlyPermuted {
            transposed,
            batch_swap: _,
        } => (false, transposed),
        MatrixLayout::HighlyPermuted => (true, false),
    };

    let (lhs_make_contiguous, lhs_transposed) = check_layout(&lhs);
    let (rhs_make_contiguous, rhs_transposed) = check_layout(&rhs);

    match (lhs_make_contiguous, rhs_make_contiguous) {
        (false, false) => matmul_cmma_ref_no_check::<R, F, F, F>(
            client,
            lhs,
            rhs,
            out,
            (lhs_transposed, rhs_transposed),
            use_cmma_if_possible,
        ),
        (false, true) => matmul_cmma_ref_no_check::<R, F, F, F>(
            client,
            lhs,
            into_contiguous::<R, F>(client, rhs).as_ref(),
            out,
            (lhs_transposed, rhs_transposed),
            use_cmma_if_possible,
        ),
        (true, false) => matmul_cmma_ref_no_check::<R, F, F, F>(
            client,
            into_contiguous::<R, F>(client, lhs).as_ref(),
            rhs,
            out,
            (lhs_transposed, rhs_transposed),
            use_cmma_if_possible,
        ),
        (true, true) => matmul_cmma_ref_no_check::<R, F, F, F>(
            client,
            into_contiguous::<R, F>(client, lhs).as_ref(),
            into_contiguous::<R, F>(client, rhs).as_ref(),
            out,
            (lhs_transposed, rhs_transposed),
            use_cmma_if_possible,
        ),
    }
}

fn matmul_cmma_ref_no_check<R: Runtime, EG: Numeric, ES: Numeric, EA: Numeric>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandleRef<'_, R>,
    rhs: TensorHandleRef<'_, R>,
    out: TensorHandleRef<'_, R>,
    transposed: (bool, bool),
    use_cmma_if_possible: bool,
) {
    let rank = lhs.strides.len();

    let m = lhs.shape[rank - 2] as u32;
    let k = lhs.shape[rank - 1] as u32;
    let n = rhs.shape[rank - 1] as u32;

    let available_vectorizations = R::supported_line_sizes();
    let lhs_line_size =
        tensor_line_size(available_vectorizations, lhs.shape, lhs.strides, rank - 1);
    let rhs_line_size =
        tensor_line_size(available_vectorizations, rhs.shape, rhs.strides, rank - 1);
    let out_line_size =
        tensor_line_size(available_vectorizations, out.shape, out.strides, rank - 1);

    let problem = MatmulProblem {
        m: m as usize,
        n: n as usize,
        k: k as usize,
        b: out.shape[..out.shape.len() - 2].to_vec(),
        lhs_layout: match transposed.0 {
            true => super::matrix::MatrixLayout::ColMajor,
            false => super::matrix::MatrixLayout::RowMajor,
        },
        rhs_layout: match transposed.1 {
            true => super::matrix::MatrixLayout::ColMajor,
            false => super::matrix::MatrixLayout::RowMajor,
        },
        lhs_line_size,
        rhs_line_size,
        out_line_size,
    };

    let cube_dim = CubeDim::new(32, 4, 1);
    let cube_count = CubeCount::Static(
        (problem.m as u32 + 63) / 64,
        (problem.n as u32 + 63) / 64,
        problem.num_batches() as u32,
    );
    let advanced_config = Default::default();

    type TmmConfig = CmmaTileMatmulConfig;
    if use_cmma_if_possible && check_cmma_availability::<R>(client).is_ok() {
        launch_typed::<R, EG, half::f16, f32, CmmaInstruction16_16_16<half::f16, f32, TmmConfig>>(
            client,
            lhs,
            rhs,
            out,
            problem,
            cube_dim,
            cube_count,
            advanced_config,
        );
    } else {
        launch_typed::<R, EG, f32, f32, PlaneMma16x16x16<f32, f32, TmmConfig>>(
            client,
            lhs,
            rhs,
            out,
            problem,
            cube_dim,
            cube_count,
            advanced_config,
        );
    }
}

fn launch_typed<
    R: Runtime,
    EG: Numeric,
    ES: Numeric,
    EA: Numeric,
    TMM: TileMatmul<ES, EA, CmmaTileMatmulConfig>,
>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandleRef<'_, R>,
    rhs: TensorHandleRef<'_, R>,
    out: TensorHandleRef<'_, R>,
    problem: MatmulProblem,
    cube_dim: CubeDim,
    cube_count: CubeCount,
    advanced_config: AdvancedConfig,
) {
    type TmmConfig = CmmaTileMatmulConfig;
    type SmmConfig = CmmaStageMatmulConfig<TmmConfig>;
    type GmmConfig = CmmaGlobalMatmulConfig<SmmConfig>;
    type BmmConfig = CmmaBatchMatmulConfig<GmmConfig>;

    type StageMatmul<TMM, EG, ES, EA> = CmmaStageMatmul<ES, EG, EA, TMM, S4x4x2, SmmConfig>;
    type GlobalMatmul<TMM, EG, ES, EA> =
        CmmaGlobalMatmul<EG, ES, StageMatmul<TMM, EG, ES, EA>, GmmConfig>;
    type BatchMatmul<TMM, EG, ES, EA> =
        CmmaBatchMatmul<EG, ES, GlobalMatmul<TMM, EG, ES, EA>, BmmConfig>;

    let config = make_cmma_config::<
        EG,
        ES,
        EA,
        TMM,
        StageMatmul<TMM, EG, ES, EA>,
        GlobalMatmul<TMM, EG, ES, EA>,
        BatchMatmul<TMM, EG, ES, EA>,
        R,
    >(&problem, &cube_dim, &cube_count, &advanced_config);

    unsafe {
        BatchMatmul::<TMM, EG, ES, EA>::launch_unchecked(
            client,
            cube_dim,
            cube_count,
            TensorArg::<R>::from_raw_parts(
                lhs.handle,
                lhs.strides,
                lhs.shape,
                problem.lhs_line_size,
            ),
            TensorArg::<R>::from_raw_parts(
                rhs.handle,
                rhs.strides,
                rhs.shape,
                problem.rhs_line_size,
            ),
            TensorArg::<R>::from_raw_parts(
                out.handle,
                out.strides,
                out.shape,
                problem.out_line_size,
            ),
            config,
        );
    }
}

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
