use core::any::TypeId;
use cubecl_core::prelude::*;

use cubecl_core::{
    client::ComputeClient, frontend::TensorHandleRef, tensor_line_size_parallel, Runtime,
};

use crate::matmul;
use crate::matmul::components::global::args::TensorInputsLaunch;
use crate::matmul::components::{
    InputRuntimeArg, MatmulLaunch, MatmulProblem, MatmulSpec, OutputRuntimeArg, SingleMatmulSpec,
};
use crate::matmul::kernels::MatmulLaunchError;
use crate::tensor::{into_contiguous, matrix_layout, MatrixLayout, TensorHandle};

use super::algorithm::{CmmaSelector, PlaneMmaSelector};
use super::config::AdvancedConfig;
use super::Algorithm;

/// Launch a matrix multiplication kernel.
///
/// Cmma will be used if enabled
/// Will fail if unavailable
pub fn launch<R: Runtime, EG: Numeric>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandle<R, EG>,
    rhs: TensorHandle<R, EG>,
    out: TensorHandle<R, EG>,
    disable_cmma: bool,
) -> Result<TensorHandle<R, EG>, MatmulLaunchError> {
    let result = launch_ref::<R, EG>(
        client,
        &lhs.as_ref(),
        &rhs.as_ref(),
        &out.as_ref(),
        disable_cmma,
    );

    match result {
        Ok(_) => Ok(out),
        Err(e) => Err(e),
    }
}

/// Launch a matrix multiplication kernel.
///
/// Cmma will be used if available and enabled,
/// otherwise it will fall back on a non-cmma implementation
pub fn launch_ref<R: Runtime, EG: Numeric>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: &TensorHandleRef<'_, R>,
    rhs: &TensorHandleRef<'_, R>,
    out: &TensorHandleRef<'_, R>,
    disable_cmma: bool,
) -> Result<(), MatmulLaunchError> {
    let check_layout = |tensor: &TensorHandleRef<'_, R>| match matrix_layout(tensor.strides) {
        MatrixLayout::Contiguous => (false, false),
        MatrixLayout::MildlyPermuted {
            transposed,
            batch_swap: _,
        } => (false, transposed),
        MatrixLayout::HighlyPermuted => (true, false),
    };

    let (lhs_make_contiguous, lhs_transposed) = check_layout(lhs);
    let (rhs_make_contiguous, rhs_transposed) = check_layout(rhs);

    match (lhs_make_contiguous, rhs_make_contiguous) {
        (false, false) => matmul_cmma_ref_no_check::<R, EG>(
            client,
            lhs,
            rhs,
            out,
            (lhs_transposed, rhs_transposed),
            disable_cmma,
        ),
        (false, true) => matmul_cmma_ref_no_check::<R, EG>(
            client,
            lhs,
            &into_contiguous::<R, EG>(client, rhs).as_ref(),
            out,
            (lhs_transposed, rhs_transposed),
            disable_cmma,
        ),
        (true, false) => matmul_cmma_ref_no_check::<R, EG>(
            client,
            &into_contiguous::<R, EG>(client, lhs).as_ref(),
            rhs,
            out,
            (lhs_transposed, rhs_transposed),
            disable_cmma,
        ),
        (true, true) => matmul_cmma_ref_no_check::<R, EG>(
            client,
            &into_contiguous::<R, EG>(client, lhs).as_ref(),
            &into_contiguous::<R, EG>(client, rhs).as_ref(),
            out,
            (lhs_transposed, rhs_transposed),
            disable_cmma,
        ),
    }
}

fn matmul_cmma_ref_no_check<R: Runtime, EG: Numeric>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: &TensorHandleRef<'_, R>,
    rhs: &TensorHandleRef<'_, R>,
    out: &TensorHandleRef<'_, R>,
    transposed: (bool, bool),
    disable_cmma: bool,
) -> Result<(), MatmulLaunchError> {
    let rank = lhs.strides.len();

    let m = lhs.shape[rank - 2] as u32;
    let k = lhs.shape[rank - 1] as u32;
    let n = rhs.shape[rank - 1] as u32;

    let lhs_line_size = tensor_line_size_parallel(
        R::line_size_elem(&EG::as_elem()),
        lhs.shape,
        lhs.strides,
        rank - 1,
    );
    let rhs_line_size = tensor_line_size_parallel(
        R::line_size_elem(&EG::as_elem()),
        rhs.shape,
        rhs.strides,
        rank - 1,
    );
    let out_line_size = tensor_line_size_parallel(
        R::line_size_elem(&EG::as_elem()),
        out.shape,
        out.strides,
        rank - 1,
    );

    let problem = MatmulProblem {
        m: m as usize,
        n: n as usize,
        k: k as usize,
        batches: (
            lhs.shape[..lhs.shape.len() - 2].to_vec(),
            rhs.shape[..rhs.shape.len() - 2].to_vec(),
        ),
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
    };

    if disable_cmma {
        PlaneMmaSelector::select_kernel::<SingleMatmulSpec<EG, EG, f32>, R>(
            client,
            TensorInputsLaunch::new(
                lhs.as_tensor_arg(lhs_line_size),
                rhs.as_tensor_arg(rhs_line_size),
            ),
            out.as_tensor_arg(out_line_size),
            problem,
        )
    } else if TypeId::of::<EG>() == TypeId::of::<half::f16>()
        || TypeId::of::<EG>() == TypeId::of::<flex32>()
    {
        CmmaSelector::select_kernel::<SingleMatmulSpec<EG, half::f16, f32>, R>(
            client,
            TensorInputsLaunch::new(
                lhs.as_tensor_arg(lhs_line_size),
                rhs.as_tensor_arg(rhs_line_size),
            ),
            out.as_tensor_arg(out_line_size),
            problem,
        )
    } else {
        CmmaSelector::select_kernel::<SingleMatmulSpec<EG, tf32, f32>, R>(
            client,
            TensorInputsLaunch::new(
                lhs.as_tensor_arg(lhs_line_size),
                rhs.as_tensor_arg(rhs_line_size),
            ),
            out.as_tensor_arg(out_line_size),
            problem,
        )
    }
}

pub(crate) fn matmul_cube_preparation<'a, MS: MatmulSpec, R: Runtime, D: Algorithm<MS>>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: InputRuntimeArg<'a, MS, R>,
    output: OutputRuntimeArg<'a, MS, R>,
    problem: MatmulProblem,
) -> Result<(), MatmulLaunchError> {
    D::check_availability::<R>(client)?;

    let cube_dim = D::cube_dim();
    let cube_count = D::cube_count(&problem);
    let advanced_config = D::advanced_config();

    launch_matmul::<MS, R, D>(
        client,
        input,
        output,
        problem,
        cube_dim,
        cube_count,
        advanced_config,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_matmul<'a, MS: MatmulSpec, R: Runtime, D: Algorithm<MS>>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: InputRuntimeArg<'a, MS, R>,
    output: OutputRuntimeArg<'a, MS, R>,
    problem: MatmulProblem,
    cube_dim: CubeDim,
    cube_count: CubeCount,
    advanced_config: AdvancedConfig,
) -> Result<(), MatmulLaunchError> {
    let config = D::make_config(&problem, &cube_dim, &cube_count, &advanced_config)?;

    unsafe {
        D::BatchMatmul::launch_unchecked::<R>(client, cube_dim, cube_count, input, output, config);
    };

    Ok(())
}
