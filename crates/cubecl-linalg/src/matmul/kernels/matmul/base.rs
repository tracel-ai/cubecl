use crate::matmul::components::{self, CompleteStageTiling, global::args::TensorInputsLaunch};
use crate::matmul::components::{
    InputRuntimeArg, MatmulConfigFactory, MatmulLaunch, MatmulProblem, MatmulSelection, MatmulSpec,
    OutputRuntimeArg, SingleMatmulSpec,
};
use crate::matmul::components::{global::args::TensorMapArgs, tile::TileMatmulFamily};
use crate::matmul::kernels::{
    MatmulAvailabilityError, MatmulLaunchError, MatmulUnimplementedError,
};
use crate::matmul::{self, components::global::args::TensorMapInputsLaunch};
use crate::tensor::{MatrixLayout, TensorHandle, into_contiguous, matrix_layout};
use core::any::TypeId;
use cubecl_core::prelude::*;
use cubecl_core::{
    Runtime, client::ComputeClient, frontend::TensorHandleRef, tensor_line_size_parallel,
};
use cubecl_std::MaybeQuantized;

use super::{Algorithm, matmul_selection, select_kernel};

/// Launch a matrix multiplication kernel.
///
/// Cmma will be used if enabled
/// Will fail if unavailable
#[allow(clippy::result_large_err)]
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
#[allow(clippy::result_large_err)]
pub fn launch_ref<R: Runtime, EG: MaybeQuantized, A: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: &TensorHandleRef<'_, R>,
    rhs: &TensorHandleRef<'_, R>,
    out: &TensorHandleRef<'_, R>,
) -> Result<(), MatmulLaunchError> {
    if EG::QUANTIZED {
        return Err(MatmulLaunchError::Unimplemented(
            MatmulUnimplementedError::Quantization,
        ));
    }

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

#[allow(clippy::result_large_err)]
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

    if EG::QUANTIZED {
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

#[allow(clippy::result_large_err)]
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
        select_kernel::<SingleMatmulSpec<i8, i8, i32>, R, A>(
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
    } else if <A::TileMatmul as TileMatmulFamily>::requires_tensor_cores()
        && tf32::is_supported(client)
    {
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
    } else if <A::TileMatmul as TileMatmulFamily>::requires_tensor_cores() {
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

#[allow(clippy::result_large_err)]
pub fn matmul_cmma_tma_ref_no_check<R: Runtime, EG: MaybeQuantized, A: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: &TensorHandleRef<'_, R>,
    rhs: &TensorHandleRef<'_, R>,
    out: &TensorHandleRef<'_, R>,
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
        lhs_layout: matmul::components::MatrixLayout::RowMajor,
        rhs_layout: matmul::components::MatrixLayout::RowMajor,
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
            ));
        }
        None => {
            return Err(MatmulLaunchError::Unavailable(
                MatmulAvailabilityError::PlaneDimUnknown,
            ));
        }
    };

    matmul_launch_kernel_tma::<R, EG, A>(
        client,
        lhs,
        rhs,
        out,
        (lhs_line_size, rhs_line_size, out_line_size),
        problem,
        plane_dim,
    )
}

#[allow(clippy::result_large_err)]
fn matmul_launch_kernel_tma<R: Runtime, EG: MaybeQuantized, A: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: &TensorHandleRef<'_, R>,
    rhs: &TensorHandleRef<'_, R>,
    out: &TensorHandleRef<'_, R>,
    (_, _, out_line_size): (u8, u8, u8),
    problem: MatmulProblem,
    plane_dim: u32,
) -> Result<(), MatmulLaunchError> {
    if TypeId::of::<EG::Numeric>() == TypeId::of::<half::f16>() {
        let selection = matmul_selection::<
            A::TileMatmul,
            SingleMatmulSpec<EG::Numeric, half::f16, f32, TensorMapArgs>,
            R,
        >(client, &problem, plane_dim);
        let stage_m = selection.tile_count.m * selection.tile_shape.m;
        let stage_n = selection.tile_count.n * selection.tile_shape.n;
        let stage_k = selection.tile_count.k * selection.tile_shape.k;
        let stage_size_lhs = match problem.lhs_layout {
            components::MatrixLayout::RowMajor => vec![1, stage_m, stage_k],
            components::MatrixLayout::ColMajor => vec![1, stage_k, stage_m],
        };
        let stage_size_rhs = match problem.rhs_layout {
            components::MatrixLayout::RowMajor => vec![1, stage_k, stage_n],
            components::MatrixLayout::ColMajor => vec![1, stage_n, stage_k],
        };
        let lhs = TensorMapArg::new(
            TensorMapFormat::Tiled {
                tile_size: stage_size_lhs,
            },
            lhs.as_tensor_arg(1),
            half::f16::as_elem_native_unchecked(),
        );
        let rhs = TensorMapArg::new(
            TensorMapFormat::Tiled {
                tile_size: stage_size_rhs,
            },
            rhs.as_tensor_arg(1),
            half::f16::as_elem_native_unchecked(),
        );
        let config_input = CompleteStageTiling {
            tile_shape: selection.tile_shape,
            tile_count: selection.tile_count,
        };

        matmul_cube_preparation::<SingleMatmulSpec<EG::Numeric, half::f16, f32, TensorMapArgs>, R, A>(
            client,
            TensorMapInputsLaunch::new(lhs, rhs),
            out.as_tensor_arg(out_line_size),
            problem,
            config_input,
            selection,
            false,
        )
    } else {
        todo!()
    }
}

#[allow(clippy::result_large_err)]
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

#[allow(clippy::too_many_arguments, clippy::result_large_err)]
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
