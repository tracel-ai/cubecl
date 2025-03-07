use crate::matmul;
use crate::matmul::components::global::args::TensorInputsLaunch;
use crate::matmul::components::tile::TileMatmulFamily;
use crate::matmul::components::{
    InputRuntimeArg, MatmulConfigFactory, MatmulLaunch, MatmulProblem, MatmulSelection, MatmulSpec,
    OutputRuntimeArg, SingleMatmulSpec,
};
use crate::matmul::kernels::{MatmulAvailabilityError, MatmulLaunchError};
use crate::tensor::{into_contiguous, matrix_layout, MatrixLayout, TensorHandle};
use core::any::TypeId;
use cubecl_core::prelude::*;
use cubecl_core::{
    client::ComputeClient, frontend::TensorHandleRef, tensor_line_size_parallel, Runtime,
};
use cubecl_std::MaybeQuantized;

use super::{select_kernel, Algorithm};

/// Launch a matrix multiplication kernel.
///
/// Cmma will be used if enabled
/// Will fail if unavailable
pub fn launch<R: Runtime, EG: MaybeQuantized, A: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandle<R, EG::Numeric>,
    rhs: TensorHandle<R, EG::Numeric>,
    out: TensorHandle<R, EG::Numeric>,
) -> Result<TensorHandle<R, EG::Numeric>, MatmulLaunchError> {
    let result = launch_ref::<R, EG, A>(client, &lhs.as_ref(), &rhs.as_ref(), &out.as_ref());

    match result {
        Ok(_) => Ok(out),
        Err(e) => Err(e),
    }
}

/// Launch a matrix multiplication kernel.
///
/// Cmma will be used if available and enabled,
/// otherwise it will fall back on a non-cmma implementation
pub fn launch_ref<R: Runtime, EG: MaybeQuantized, A: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: &TensorHandleRef<'_, R>,
    rhs: &TensorHandleRef<'_, R>,
    out: &TensorHandleRef<'_, R>,
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
        (false, false) => matmul_cmma_ref_no_check::<R, EG, A>(
            client,
            lhs,
            rhs,
            out,
            (lhs_transposed, rhs_transposed),
        ),
        (false, true) => matmul_cmma_ref_no_check::<R, EG, A>(
            client,
            lhs,
            &into_contiguous::<R, EG::Numeric>(client, rhs).as_ref(),
            out,
            (lhs_transposed, rhs_transposed),
        ),
        (true, false) => matmul_cmma_ref_no_check::<R, EG, A>(
            client,
            &into_contiguous::<R, EG::Numeric>(client, lhs).as_ref(),
            rhs,
            out,
            (lhs_transposed, rhs_transposed),
        ),
        (true, true) => matmul_cmma_ref_no_check::<R, EG, A>(
            client,
            &into_contiguous::<R, EG::Numeric>(client, lhs).as_ref(),
            &into_contiguous::<R, EG::Numeric>(client, rhs).as_ref(),
            out,
            (lhs_transposed, rhs_transposed),
        ),
    }
}

fn matmul_cmma_ref_no_check<R: Runtime, EG: MaybeQuantized, A: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: &TensorHandleRef<'_, R>,
    rhs: &TensorHandleRef<'_, R>,
    out: &TensorHandleRef<'_, R>,
    transposed: (bool, bool),
) -> Result<(), MatmulLaunchError> {
    let rank = lhs.strides.len();
    let eg_elem = EG::Numeric::as_elem_native().expect("To be a native type");

    let m = lhs.shape[rank - 2] as u32;
    let k = lhs.shape[rank - 1] as u32;
    let n = rhs.shape[rank - 1] as u32;

    let lhs_line_size = tensor_line_size_parallel(
        R::line_size_elem(&eg_elem),
        lhs.shape,
        lhs.strides,
        rank - 1,
    );
    let rhs_line_size = tensor_line_size_parallel(
        R::line_size_elem(&eg_elem),
        rhs.shape,
        rhs.strides,
        rank - 1,
    );
    let out_line_size = tensor_line_size_parallel(
        R::line_size_elem(&eg_elem),
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
        // TODO consider a quantized field for MatmulProblem
    };

    let plane_size = client
        .properties()
        .hardware_properties()
        .defined_plane_size();

    let plane_dim = match plane_size {
        Some(32) | Some(64) => plane_size.expect("32 or 64"),
        Some(plane_dim) => {
            return Err(MatmulLaunchError::Unavailable(
                MatmulAvailabilityError::PlaneDimUnsupported { plane_dim },
            ))
        }
        None => {
            return Err(MatmulLaunchError::Unavailable(
                MatmulAvailabilityError::PlaneDimUnknown,
            ))
        }
    };

    matmul_launch_kernel::<R, EG, A>(
        client,
        lhs,
        rhs,
        out,
        (lhs_line_size, rhs_line_size, out_line_size),
        problem,
        plane_dim,
    )
}

fn matmul_launch_kernel<R: Runtime, EG: MaybeQuantized, A: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: &TensorHandleRef<'_, R>,
    rhs: &TensorHandleRef<'_, R>,
    out: &TensorHandleRef<'_, R>,
    (lhs_line_size, rhs_line_size, out_line_size): (u8, u8, u8),
    problem: MatmulProblem,
    plane_dim: u32,
) -> Result<(), MatmulLaunchError> {
    if EG::QUANTIZED {
        select_kernel::<SingleMatmulSpec<u8, u16, i32>, R, A>(
            client,
            TensorInputsLaunch::new(
                lhs.as_tensor_arg(lhs_line_size),
                rhs.as_tensor_arg(rhs_line_size),
            ),
            out.as_tensor_arg(out_line_size),
            problem,
            plane_dim,
            true,
        )
    } else if TypeId::of::<EG::Numeric>() == TypeId::of::<half::f16>()
        || TypeId::of::<EG::Numeric>() == TypeId::of::<flex32>()
    {
        select_kernel::<SingleMatmulSpec<EG::Numeric, half::f16, f32>, R, A>(
            client,
            TensorInputsLaunch::new(
                lhs.as_tensor_arg(lhs_line_size),
                rhs.as_tensor_arg(rhs_line_size),
            ),
            out.as_tensor_arg(out_line_size),
            problem,
            plane_dim,
            false,
        )
    } else if TypeId::of::<EG::Numeric>() == TypeId::of::<half::bf16>() {
        select_kernel::<SingleMatmulSpec<EG::Numeric, half::bf16, f32>, R, A>(
            client,
            TensorInputsLaunch::new(
                lhs.as_tensor_arg(lhs_line_size),
                rhs.as_tensor_arg(rhs_line_size),
            ),
            out.as_tensor_arg(out_line_size),
            problem,
            plane_dim,
            false,
        )
    } else if <A::TileMatmul as TileMatmulFamily>::requires_tensor_cores() {
        select_kernel::<SingleMatmulSpec<EG::Numeric, tf32, f32>, R, A>(
            client,
            TensorInputsLaunch::new(
                lhs.as_tensor_arg(lhs_line_size),
                rhs.as_tensor_arg(rhs_line_size),
            ),
            out.as_tensor_arg(out_line_size),
            problem,
            plane_dim,
            false,
        )
    } else {
        select_kernel::<SingleMatmulSpec<EG::Numeric, EG::Numeric, f32>, R, A>(
            client,
            TensorInputsLaunch::new(
                lhs.as_tensor_arg(lhs_line_size),
                rhs.as_tensor_arg(rhs_line_size),
            ),
            out.as_tensor_arg(out_line_size),
            problem,
            plane_dim,
            false,
        )
    }
}

pub(crate) fn matmul_cube_preparation<'a, MS: MatmulSpec, R: Runtime, A: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: InputRuntimeArg<'a, MS, R>,
    output: OutputRuntimeArg<'a, MS, R>,
    problem: MatmulProblem,
    config_input: <A::BatchMatmul as MatmulConfigFactory>::Input,
    selection: MatmulSelection,
    quantized: bool,
) -> Result<(), MatmulLaunchError> {
    let cube_dim = A::cube_dim(&selection);
    let cube_count = A::cube_count(&selection, &problem);

    launch_matmul::<MS, R, A>(
        client,
        input,
        output,
        problem,
        cube_dim,
        cube_count,
        config_input,
        quantized,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_matmul<'a, MS: MatmulSpec, R: Runtime, D: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: InputRuntimeArg<'a, MS, R>,
    output: OutputRuntimeArg<'a, MS, R>,
    problem: MatmulProblem,
    cube_dim: CubeDim,
    cube_count: CubeCount,
    config_input: <D::BatchMatmul as MatmulConfigFactory>::Input,
    quantized: bool,
) -> Result<(), MatmulLaunchError> {
    let config = D::make_config(config_input, &problem, &cube_dim, &cube_count, quantized)?;
    D::check_availability::<R, (MS::EG, MS::ES, MS::EA)>(client, &config)?;

    unsafe {
        D::BatchMatmul::launch_unchecked::<MS, R>(
            client, cube_dim, cube_count, input, output, config,
        );
    };

    Ok(())
}
