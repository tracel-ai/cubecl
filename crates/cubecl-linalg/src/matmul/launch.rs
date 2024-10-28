use super::cmma_matmul::batch::{CmmaBatchMatmul, CmmaBatchMatmulConfig};
use super::cmma_matmul::global::{CmmaGlobalMatmul, CmmaGlobalMatmulConfig};
use super::cmma_matmul::stage::{CmmaStageMatmul, CmmaStageMatmulConfig, S4x4x2};
use super::cmma_matmul::tile::{
    check_cmma_availability, CmmaInstruction16_16_16, CmmaTileMatmulConfig, PlaneMma16x16x16,
};
use super::matmul_tile::TileMatmul;
use super::problem::MatmulProblem;
use super::tests::{make_cmma_config, AdvancedConfig};
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
) -> TensorHandle<R, F> {
    matmul_cmma_ref::<R, F>(client, lhs.as_ref(), rhs.as_ref(), out.as_ref());
    out
}

pub fn matmul_cmma_ref<R: Runtime, F: Float>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandleRef<'_, R>,
    rhs: TensorHandleRef<'_, R>,
    out: TensorHandleRef<'_, R>,
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
        ),
        (false, true) => matmul_cmma_ref_no_check::<R, F, F, F>(
            client,
            lhs,
            into_contiguous::<R, F>(client, rhs).as_ref(),
            out,
            (lhs_transposed, rhs_transposed),
        ),
        (true, false) => matmul_cmma_ref_no_check::<R, F, F, F>(
            client,
            into_contiguous::<R, F>(client, lhs).as_ref(),
            rhs,
            out,
            (lhs_transposed, rhs_transposed),
        ),
        (true, true) => matmul_cmma_ref_no_check::<R, F, F, F>(
            client,
            into_contiguous::<R, F>(client, lhs).as_ref(),
            into_contiguous::<R, F>(client, rhs).as_ref(),
            out,
            (lhs_transposed, rhs_transposed),
        ),
    }
}

fn matmul_cmma_ref_no_check<R: Runtime, EG: Numeric, ES: Numeric, EA: Numeric>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandleRef<'_, R>,
    rhs: TensorHandleRef<'_, R>,
    out: TensorHandleRef<'_, R>,
    transposed: (bool, bool),
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

    let cube_dim = CubeDim::new(32, 1, 1);
    let cube_count = CubeCount::Static(1, 1, 1);
    let advanced_config = Default::default();

    type TmmConfig = CmmaTileMatmulConfig;
    if check_cmma_availability::<R>(client).is_ok() {
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
