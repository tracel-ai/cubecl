use std::marker::PhantomData;

use cubecl_core::prelude::*;

use cubecl_core::{
    client::ComputeClient,
    frontend::{TensorArg, TensorHandleRef},
    tensor_line_size, Runtime,
};

use crate::matmul;
use crate::matmul::components::{batch, global, stage, MatmulLaunch, MatmulProblem};
use crate::matmul::kernels::cmma_matmul::config::make_cmma_config;
use crate::tensor::{into_contiguous, matrix_layout, MatrixLayout, TensorHandle};

use super::config::AdvancedConfig;
use super::dispatch::{
    check_availability, CmmaLaunchDispatch, MatmulLaunchDispatch, PlaneMmaLaunchDispatch,
};

/// Launch a matrix multiplication kernel.
///
/// Cmma will be used if available and enabled,
/// otherwise it will fall back on a non-cmma implementation
pub fn launch_ref<R: Runtime, EG: Numeric>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandleRef<'_, R>,
    rhs: TensorHandleRef<'_, R>,
    out: TensorHandleRef<'_, R>,
    disable_cmma: bool,
) {
    if !disable_cmma && check_availability::<CmmaLaunchDispatch, R>(client).is_ok() {
        matmul_cmma_ref::<R, EG, CmmaLaunchDispatch>(client, lhs, rhs, out);
    } else {
        matmul_cmma_ref::<R, EG, PlaneMmaLaunchDispatch>(client, lhs, rhs, out);
    }
}

/// Launch a matrix multiplication kernel.
///
/// Cmma will be used if available and enabled,
/// otherwise it will fall back on a non-cmma implementation
pub fn launch<R: Runtime, EG: Numeric>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandle<R, EG>,
    rhs: TensorHandle<R, EG>,
    out: TensorHandle<R, EG>,
    disable_cmma: bool,
) -> TensorHandle<R, EG> {
    launch_ref::<R, EG>(
        client,
        lhs.as_ref(),
        rhs.as_ref(),
        out.as_ref(),
        disable_cmma,
    );
    out
}

fn matmul_cmma_ref<R: Runtime, E: Numeric, D: MatmulLaunchDispatch>(
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
            true => matmul::components::MatrixLayout::ColMajor,
            false => matmul::components::MatrixLayout::RowMajor,
        },
        rhs_layout: match transposed.1 {
            true => matmul::components::MatrixLayout::ColMajor,
            false => matmul::components::MatrixLayout::RowMajor,
        },
        lhs_line_size,
        rhs_line_size,
        out_line_size,
        _element: PhantomData,
    };

    let cube_dim = D::cube_dim();
    let cube_count = D::cube_count(&problem);

    let advanced_config = Default::default();

    launch_matmul::<R, EG, D>(
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

#[allow(clippy::too_many_arguments)]
fn launch_matmul<R: Runtime, EG: Numeric, D: MatmulLaunchDispatch>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandleRef<'_, R>,
    rhs: TensorHandleRef<'_, R>,
    out: TensorHandleRef<'_, R>,
    problem: MatmulProblem<EG>,
    cube_dim: CubeDim,
    cube_count: CubeCount,
    advanced_config: AdvancedConfig,
) {
    let config = make_cmma_config::<EG, D>(&problem, &cube_dim, &cube_count, &advanced_config);

    type StageMatmul<TMM, CSS, EG, ES, EA> = stage::row_accumulate::Matmul<ES, EG, EA, TMM, CSS>;
    type GlobalMatmul<TMM, CSS, EG, ES, EA> =
        global::homogeneous::Matmul<EG, ES, StageMatmul<TMM, CSS, EG, ES, EA>>;

    unsafe {
        batch::one_to_one::Matmul::<
            EG,
            D::ElementInput,
            GlobalMatmul<D::TileMatmul, D::StageSize, EG, D::ElementInput, D::ElementAccumulator>,
        >::launch_unchecked::<R>(
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
