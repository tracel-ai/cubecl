use std::marker::PhantomData;

use cubecl_core::prelude::*;

use cubecl_core::{
    client::ComputeClient,
    frontend::{TensorArg, TensorHandleRef},
    tensor_line_size, Runtime,
};

use crate::matmul::components::batch::OneToOneBatchMatmul;
use crate::matmul::components::cmma_matmul::global::HomogeneousGlobalMatmul;
use crate::matmul::components::cmma_matmul::launch::{
    make_cmma_config, CmmaBmmConfig, CmmaGmmConfig, CmmaSmmConfig,
};
use crate::matmul::components::problem::MatmulProblem;
use crate::matmul::components::stage::{RowAccumulateStageMatmul, StageSize};
use crate::matmul::components::tile::{TileConfig, TileMatmul};
use crate::matmul::components::{matrix, MatmulLaunch};
use crate::tensor::{into_contiguous, matrix_layout, MatrixLayout};

use super::{AdvancedConfig, MatmulLaunchDispatch};

/// Launches a matmul on tensor handles
pub fn matmul_cmma_ref<R: Runtime, E: Numeric, D: MatmulLaunchDispatch>(
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
        (false, false) => matmul_cmma_ref_no_check::<R, E, D>(
            client,
            lhs,
            rhs,
            out,
            (lhs_transposed, rhs_transposed),
        ),
        (false, true) => matmul_cmma_ref_no_check::<R, E, D>(
            client,
            lhs,
            into_contiguous::<R, E>(client, rhs).as_ref(),
            out,
            (lhs_transposed, rhs_transposed),
        ),
        (true, false) => matmul_cmma_ref_no_check::<R, E, D>(
            client,
            into_contiguous::<R, E>(client, lhs).as_ref(),
            rhs,
            out,
            (lhs_transposed, rhs_transposed),
        ),
        (true, true) => matmul_cmma_ref_no_check::<R, E, D>(
            client,
            into_contiguous::<R, E>(client, lhs).as_ref(),
            into_contiguous::<R, E>(client, rhs).as_ref(),
            out,
            (lhs_transposed, rhs_transposed),
        ),
    }
}

fn matmul_cmma_ref_no_check<R: Runtime, EG: Numeric, D: MatmulLaunchDispatch>(
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

    let problem = MatmulProblem::<EG> {
        m: m as usize,
        n: n as usize,
        k: k as usize,
        batches: out.shape[..out.shape.len() - 2].to_vec(),
        lhs_layout: match transposed.0 {
            true => matrix::MatrixLayout::ColMajor,
            false => matrix::MatrixLayout::RowMajor,
        },
        rhs_layout: match transposed.1 {
            true => matrix::MatrixLayout::ColMajor,
            false => matrix::MatrixLayout::RowMajor,
        },
        lhs_line_size,
        rhs_line_size,
        out_line_size,
        _element: PhantomData,
    };

    let cube_dim = D::cube_dim();
    let cube_count = D::cube_count(&problem);

    let advanced_config = Default::default();

    launch_matmul::<R, EG, D::ElementInput, D::ElementAccumulator, D::TileMatmul, D::StageSize>(
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

fn launch_matmul<
    R: Runtime,
    EG: Numeric,
    ES: Numeric,
    EA: Numeric,
    TMM: TileMatmul<ES, EA, TileConfig>,
    CSS: StageSize,
>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandleRef<'_, R>,
    rhs: TensorHandleRef<'_, R>,
    out: TensorHandleRef<'_, R>,
    problem: MatmulProblem<EG>,
    cube_dim: CubeDim,
    cube_count: CubeCount,
    advanced_config: AdvancedConfig,
) {
    type StageMatmul<TMM, CSS, EG, ES, EA> =
        RowAccumulateStageMatmul<ES, EG, EA, TMM, CSS, CmmaSmmConfig>;
    type GlobalMatmul<TMM, CSS, EG, ES, EA> =
        HomogeneousGlobalMatmul<EG, ES, StageMatmul<TMM, CSS, EG, ES, EA>, CmmaGmmConfig>;
    type BatchMatmul<TMM, CSS, EG, ES, EA> =
        OneToOneBatchMatmul<EG, ES, GlobalMatmul<TMM, CSS, EG, ES, EA>, CmmaBmmConfig>;

    let config = make_cmma_config::<
        EG,
        ES,
        EA,
        TMM,
        StageMatmul<TMM, CSS, EG, ES, EA>,
        GlobalMatmul<TMM, CSS, EG, ES, EA>,
        BatchMatmul<TMM, CSS, EG, ES, EA>,
        R,
    >(&problem, &cube_dim, &cube_count, &advanced_config);

    unsafe {
        BatchMatmul::<TMM, CSS, EG, ES, EA>::launch_unchecked(
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
