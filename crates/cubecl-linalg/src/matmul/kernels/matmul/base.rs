use crate::matmul::components::{
    InputRuntimeArg, MatmulConfigFactory, MatmulLaunch, MatmulPrecision, MatmulProblem,
    MatmulSelection, MatmulSpec, MatrixLayout, OutputRuntimeArg, ReplaceES,
};
use crate::matmul::components::{global::args::TensorMapArgs, tile::TileMatmulFamily};
use crate::matmul::kernels::{MatmulAvailabilityError, MatmulLaunchError};
use crate::matmul::{self};
use crate::tensor::into_contiguous_pitched;
use crate::tensor::{MatrixBatchLayout, TensorHandle, matrix_batch_layout};
use core::any::TypeId;
use cubecl_core::prelude::*;
use cubecl_core::{
    Runtime, client::ComputeClient, frontend::TensorHandleRef, tensor_line_size_parallel,
};

use super::{Algorithm, select_kernel_concrete};

/// Launch a matrix multiplication kernel.
///
/// Cmma will be used if enabled
/// Will fail if unavailable
#[allow(clippy::result_large_err)]
pub fn launch<R: Runtime, MP: MatmulPrecision, A: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandle<R, MP::EI>,
    rhs: TensorHandle<R, MP::EI>,
    out: TensorHandle<R, MP::EO>,
) -> Result<TensorHandle<R, MP::EO>, MatmulLaunchError> {
    let result = launch_ref::<R, MP, A>(client, &lhs.as_ref(), &rhs.as_ref(), &out.as_ref());

    match result {
        Ok(_) => Ok(out),
        Err(e) => Err(e),
    }
}

/// Launch a matrix multiplication kernel.
///
/// Cmma will be used if available and enabled,
/// otherwise it will fall back on a non-cmma implementation
#[allow(clippy::result_large_err)]
pub fn launch_ref<R: Runtime, MP: MatmulPrecision, A: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: &TensorHandleRef<'_, R>,
    rhs: &TensorHandleRef<'_, R>,
    out: &TensorHandleRef<'_, R>,
) -> Result<(), MatmulLaunchError> {
    let check_layout = |tensor: &TensorHandleRef<'_, R>| match matrix_batch_layout(tensor.strides) {
        MatrixBatchLayout::Contiguous => (false, false),
        MatrixBatchLayout::MildlyPermuted {
            transposed,
            batch_swap: _,
        } => (false, transposed),
        MatrixBatchLayout::HighlyPermuted => (true, false),
    };

    let (lhs_make_contiguous, lhs_transposed) = check_layout(lhs);
    let (rhs_make_contiguous, rhs_transposed) = check_layout(rhs);

    match (lhs_make_contiguous, rhs_make_contiguous) {
        (false, false) => matmul_cmma_ref_no_check::<R, MP, A>(
            client,
            lhs,
            rhs,
            out,
            (lhs_transposed, rhs_transposed),
        ),
        (false, true) => matmul_cmma_ref_no_check::<R, MP, A>(
            client,
            lhs,
            &into_contiguous_pitched::<R, MP::EI>(client, rhs).as_ref(),
            out,
            (lhs_transposed, rhs_transposed),
        ),
        (true, false) => matmul_cmma_ref_no_check::<R, MP, A>(
            client,
            &into_contiguous_pitched::<R, MP::EI>(client, lhs).as_ref(),
            rhs,
            out,
            (lhs_transposed, rhs_transposed),
        ),
        (true, true) => matmul_cmma_ref_no_check::<R, MP, A>(
            client,
            &into_contiguous_pitched::<R, MP::EI>(client, lhs).as_ref(),
            &into_contiguous_pitched::<R, MP::EI>(client, rhs).as_ref(),
            out,
            (lhs_transposed, rhs_transposed),
        ),
    }
}

#[allow(clippy::result_large_err)]
fn matmul_cmma_ref_no_check<R: Runtime, MP: MatmulPrecision, A: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: &TensorHandleRef<'_, R>,
    rhs: &TensorHandleRef<'_, R>,
    out: &TensorHandleRef<'_, R>,
    transposed: (bool, bool),
) -> Result<(), MatmulLaunchError> {
    let rank = lhs.strides.len();
    let ei_elem = MP::EI::as_elem_native().expect("To be a native type");
    let eo_elem = MP::EO::as_elem_native().expect("To be a native type");

    let m = lhs.shape[rank - 2] as u32;
    let k = lhs.shape[rank - 1] as u32;
    let n = rhs.shape[rank - 1] as u32;

    let lhs_layout = match transposed.0 {
        true => MatrixLayout::ColMajor,
        false => MatrixLayout::RowMajor,
    };

    let rhs_layout = match transposed.1 {
        true => MatrixLayout::ColMajor,
        false => MatrixLayout::RowMajor,
    };

    let lhs_line_size = tensor_line_size_parallel(
        R::line_size_elem(&ei_elem),
        lhs.shape,
        lhs.strides,
        match lhs_layout {
            MatrixLayout::RowMajor => rank - 1,
            MatrixLayout::ColMajor => rank - 2,
        },
    );
    let rhs_line_size = tensor_line_size_parallel(
        R::line_size_elem(&ei_elem),
        rhs.shape,
        rhs.strides,
        match rhs_layout {
            MatrixLayout::RowMajor => rank - 1,
            MatrixLayout::ColMajor => rank - 2,
        },
    );
    let out_line_size = tensor_line_size_parallel(
        R::line_size_elem(&eo_elem),
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
        lhs_layout,
        rhs_layout,
        lhs_line_size,
        rhs_line_size,
        out_line_size,
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
            ));
        }
        None => {
            return Err(MatmulLaunchError::Unavailable(
                MatmulAvailabilityError::PlaneDimUnknown,
            ));
        }
    };

    if MP::QUANTIZED {
        let mut batch_count_lhs = 1;
        let mut batch_count_rhs = 1;
        for axis in 0..rank - 2 {
            batch_count_lhs *= lhs.shape[axis];
            batch_count_rhs *= rhs.shape[axis];
        }
        if batch_count_lhs != batch_count_rhs {
            panic!("broadcast is not supported yet with quantization");
        }
    }

    matmul_launch_kernel::<R, MP, A>(client, lhs, rhs, out, problem, plane_dim)
}

#[allow(clippy::result_large_err)]
fn matmul_launch_kernel<R: Runtime, MP: MatmulPrecision, A: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: &TensorHandleRef<'_, R>,
    rhs: &TensorHandleRef<'_, R>,
    out: &TensorHandleRef<'_, R>,
    problem: MatmulProblem,
    plane_dim: u32,
) -> Result<(), MatmulLaunchError> {
    if <A::TileMatmul as TileMatmulFamily>::requires_tensor_cores()
        && TypeId::of::<MP::EI>() == TypeId::of::<f32>()
    {
        if tf32::is_supported(client) {
            select_kernel_concrete::<ReplaceES<MP, tf32>, R, A>(
                client, lhs, rhs, out, problem, plane_dim,
            )
        } else {
            select_kernel_concrete::<ReplaceES<MP, half::f16>, R, A>(
                client, lhs, rhs, out, problem, plane_dim,
            )
        }
    } else {
        select_kernel_concrete::<MP, R, A>(client, lhs, rhs, out, problem, plane_dim)
    }
}

#[allow(clippy::result_large_err)]
pub fn matmul_cmma_tma_ref_no_check<R: Runtime, MP: MatmulPrecision, A: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: &TensorHandleRef<'_, R>,
    rhs: &TensorHandleRef<'_, R>,
    out: &TensorHandleRef<'_, R>,
    transposed: (bool, bool),
) -> Result<(), MatmulLaunchError> {
    let rank = lhs.strides.len();
    let eo_elem = MP::EO::as_elem_native().expect("To be a native type");

    let m = lhs.shape[rank - 2] as u32;
    let k = lhs.shape[rank - 1] as u32;
    let n = rhs.shape[rank - 1] as u32;

    let lhs_layout = match transposed.0 {
        true => matmul::components::MatrixLayout::ColMajor,
        false => matmul::components::MatrixLayout::RowMajor,
    };
    let rhs_layout = match transposed.1 {
        true => matmul::components::MatrixLayout::ColMajor,
        false => matmul::components::MatrixLayout::RowMajor,
    };

    let out_line_size = tensor_line_size_parallel(
        R::line_size_elem(&eo_elem),
        out.shape,
        out.strides,
        rank - 1,
    );

    let batch_lhs: usize = lhs.shape[..lhs.shape.len() - 2].iter().product();
    let batch_rhs: usize = rhs.shape[..rhs.shape.len() - 2].iter().product();

    let problem = MatmulProblem {
        m: m as usize,
        n: n as usize,
        k: k as usize,
        batches: ([batch_lhs].to_vec(), [batch_rhs].to_vec()),
        lhs_layout,
        rhs_layout,
        // No point vectorizing, since we never deal with individual values when using TMA
        lhs_line_size: 1,
        rhs_line_size: 1,
        out_line_size,
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
            ));
        }
        None => {
            return Err(MatmulLaunchError::Unavailable(
                MatmulAvailabilityError::PlaneDimUnknown,
            ));
        }
    };

    if TypeId::of::<MP::EI>() == TypeId::of::<f32>() {
        select_kernel_concrete::<(ReplaceES<MP, tf32>, TensorMapArgs), R, A>(
            client, lhs, rhs, out, problem, plane_dim,
        )
    } else {
        select_kernel_concrete::<(MP, TensorMapArgs), R, A>(
            client, lhs, rhs, out, problem, plane_dim,
        )
    }
}

#[allow(clippy::result_large_err)]
pub fn matmul_cube_preparation<'a, MS: MatmulSpec, R: Runtime, A: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: InputRuntimeArg<'a, MS, R>,
    output: OutputRuntimeArg<'a, MS, R>,
    problem: MatmulProblem,
    config_input: <A::BatchMatmul as MatmulConfigFactory>::Input,
    selection: MatmulSelection,
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
    )
}

#[allow(clippy::too_many_arguments, clippy::result_large_err)]
fn launch_matmul<'a, MS: MatmulSpec, R: Runtime, D: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: InputRuntimeArg<'a, MS, R>,
    output: OutputRuntimeArg<'a, MS, R>,
    problem: MatmulProblem,
    cube_dim: CubeDim,
    cube_count: CubeCount,
    config_input: <D::BatchMatmul as MatmulConfigFactory>::Input,
) -> Result<(), MatmulLaunchError> {
    let config = D::make_config(
        config_input,
        &problem,
        &cube_dim,
        &cube_count,
        MS::Precision::QUANTIZED,
    )?;
    D::check_availability::<R, MS::Precision>(client, &config)?;

    unsafe {
        D::BatchMatmul::launch_unchecked::<MS, R>(
            client,
            cube_dim,
            cube_count,
            input,
            output,
            ScalarArg::new(problem.k as u32),
            config,
        );
    };

    Ok(())
}
