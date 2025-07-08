use crate::components::batch::{BatchMatmulFamily, CubeCountInputArgs};
use crate::components::{
    AvailableLineSizes, InputRuntimeArg, MatmulLineSizes, MatmulPrecision, MatmulProblem,
    MatmulSpec, MatrixLayout, OutputRuntimeArg, ReplaceES,
};
use crate::components::{global::args::TensorMapArgs, tile::TileMatmulFamily};
use crate::kernels::layered::selector::{MatmulSelection, launch_kernel_concrete};
use crate::kernels::{MatmulAvailabilityError, MatmulSetupError};
use core::any::TypeId;
use cubecl_core::{Feature, prelude::*, try_tensor_line_size_parallel};
use cubecl_core::{Runtime, client::ComputeClient, frontend::TensorHandleRef};
use cubecl_std::tensor::{
    MatrixBatchLayout, TensorHandle, into_contiguous_pitched, matrix_batch_layout,
};

use super::Algorithm;

#[derive(Debug, Clone)]
pub enum Selection<S> {
    Forced(MatmulSelection),
    Inferred(S),
}

impl<S: Default + Clone> Selection<S> {
    pub fn maybe_forced_default(s: &Option<MatmulSelection>) -> Self {
        s.as_ref()
            .map(|s| Self::Forced(s.clone()))
            .unwrap_or_default()
    }
    pub fn maybe_forced_or(s: &Option<MatmulSelection>, args: &S) -> Self {
        s.as_ref()
            .map(|s| Self::Forced(s.clone()))
            .unwrap_or_else(|| Self::Inferred(args.clone()))
    }
}

impl<S: Default> Default for Selection<S> {
    fn default() -> Self {
        Self::Inferred(Default::default())
    }
}

/// Launch a matrix multiplication kernel.
///
/// Cmma will be used if enabled
/// Will fail if unavailable
#[allow(clippy::result_large_err)]
pub fn launch<R: Runtime, MP: MatmulPrecision, A: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandle<R, MP::EI>,
    lhs_scale: Option<TensorHandle<R, f32>>,
    rhs: TensorHandle<R, MP::EI>,
    rhs_scale: Option<TensorHandle<R, f32>>,
    out: TensorHandle<R, MP::EO>,
    selection: &Selection<A::SelectionArgs>,
) -> Result<TensorHandle<R, MP::EO>, MatmulSetupError> {
    let result = launch_ref::<R, MP, A>(
        client,
        &lhs.as_ref(),
        &lhs_scale.as_ref().map(|it| it.as_ref()),
        &rhs.as_ref(),
        &rhs_scale.as_ref().map(|it| it.as_ref()),
        &out.as_ref(),
        selection,
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
#[allow(clippy::result_large_err)]
pub fn launch_ref<R: Runtime, MP: MatmulPrecision, A: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: &TensorHandleRef<'_, R>,
    lhs_scale: &Option<TensorHandleRef<'_, R>>,
    rhs: &TensorHandleRef<'_, R>,
    rhs_scale: &Option<TensorHandleRef<'_, R>>,
    out: &TensorHandleRef<'_, R>,
    selection: &Selection<A::SelectionArgs>,
) -> Result<(), MatmulSetupError> {
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
        (false, false) => launch_inner_ref::<R, MP, A>(
            client,
            lhs,
            lhs_scale,
            rhs,
            rhs_scale,
            out,
            (lhs_transposed, rhs_transposed),
            selection,
        ),
        (false, true) => launch_inner_ref::<R, MP, A>(
            client,
            lhs,
            lhs_scale,
            &into_contiguous_pitched::<R, MP::EI>(client, rhs).as_ref(),
            rhs_scale,
            out,
            (lhs_transposed, rhs_transposed),
            selection,
        ),
        (true, false) => launch_inner_ref::<R, MP, A>(
            client,
            &into_contiguous_pitched::<R, MP::EI>(client, lhs).as_ref(),
            lhs_scale,
            rhs,
            rhs_scale,
            out,
            (lhs_transposed, rhs_transposed),
            selection,
        ),
        (true, true) => launch_inner_ref::<R, MP, A>(
            client,
            &into_contiguous_pitched::<R, MP::EI>(client, lhs).as_ref(),
            lhs_scale,
            &into_contiguous_pitched::<R, MP::EI>(client, rhs).as_ref(),
            rhs_scale,
            out,
            (lhs_transposed, rhs_transposed),
            selection,
        ),
    }
}

#[allow(clippy::result_large_err, clippy::too_many_arguments)]
fn launch_inner_ref<R: Runtime, MP: MatmulPrecision, A: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: &TensorHandleRef<'_, R>,
    lhs_scale: &Option<TensorHandleRef<'_, R>>,
    rhs: &TensorHandleRef<'_, R>,
    rhs_scale: &Option<TensorHandleRef<'_, R>>,
    out: &TensorHandleRef<'_, R>,
    transposed: (bool, bool),
    selection: &Selection<A::SelectionArgs>,
) -> Result<(), MatmulSetupError> {
    let rank = lhs.strides.len();
    let ei_elem = MP::EI::as_elem_native().expect("To be a native type");
    let eo_elem = MP::EO::as_elem_native().expect("To be a native type");

    // This is mostly to check that i8 are supported for quantization.
    if !client.properties().feature_enabled(Feature::Type(ei_elem))
        || !client.properties().feature_enabled(Feature::Type(eo_elem))
    {
        return Err(MatmulSetupError::Unavailable(
            MatmulAvailabilityError::TypesUnavailable {
                input: ei_elem,
                output: eo_elem,
            },
        ));
    }

    if MP::QUANTIZED
        && !client
            .properties()
            .feature_enabled(cubecl_core::Feature::DynamicLineSize)
    {
        return Err(MatmulSetupError::Unavailable(
            MatmulAvailabilityError::DynamicLineSizeUnavailable,
        ));
    }

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
    };

    let line_sizes = AvailableLineSizes::from_elem_types::<R>(&ei_elem, &eo_elem);
    let line_sizes = A::filter_line_sizes(line_sizes);
    let line_sizes = line_sizes
        .filter_lhs_with_tensor(lhs.strides, lhs.shape, problem.lhs_layout)
        .filter_rhs_with_tensor(rhs.strides, rhs.shape, problem.rhs_layout)
        .filter_out_with_tensor(out.strides, out.shape)
        .pick_max()?;

    let fix_plane_dim = |plane_dim: u32| {
        // Sometimes the GPU doesn't support plane instructions and doesn't report the
        // plane size, but we can still execute algorithms that don't use plane instructions.
        //
        // In this case, we set a plane size for the selector to work, defaulting to 32 as it
        // is a common plane size.
        if plane_dim == 0 { 32 } else { plane_dim }
    };

    let plane_dim = fix_plane_dim(A::select_plane_dim::<R>(client));

    launch_inner_ref_fix_dtype::<R, MP, A>(
        client, lhs, lhs_scale, rhs, rhs_scale, out, problem, line_sizes, plane_dim, selection,
    )
}

#[allow(clippy::result_large_err, clippy::too_many_arguments)]
fn launch_inner_ref_fix_dtype<R: Runtime, MP: MatmulPrecision, A: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: &TensorHandleRef<'_, R>,
    lhs_scale: &Option<TensorHandleRef<'_, R>>,
    rhs: &TensorHandleRef<'_, R>,
    rhs_scale: &Option<TensorHandleRef<'_, R>>,
    out: &TensorHandleRef<'_, R>,
    problem: MatmulProblem,
    line_sizes: MatmulLineSizes,
    plane_dim: u32,
    selection: &Selection<A::SelectionArgs>,
) -> Result<(), MatmulSetupError> {
    if <A::TileMatmul as TileMatmulFamily>::requires_tensor_cores()
        && TypeId::of::<MP::ES>() == TypeId::of::<f32>()
        && tf32::is_supported(client)
    {
        launch_kernel_concrete::<ReplaceES<MP, tf32>, R, A>(
            client, lhs, lhs_scale, rhs, rhs_scale, out, problem, line_sizes, plane_dim, selection,
        )
    } else {
        launch_kernel_concrete::<MP, R, A>(
            client, lhs, lhs_scale, rhs, rhs_scale, out, problem, line_sizes, plane_dim, selection,
        )
    }
}

#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn matmul_cmma_tma_ref_no_check<R: Runtime, MP: MatmulPrecision, A: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: &TensorHandleRef<'_, R>,
    lhs_scale: &Option<TensorHandleRef<'_, R>>,
    rhs: &TensorHandleRef<'_, R>,
    rhs_scale: &Option<TensorHandleRef<'_, R>>,
    out: &TensorHandleRef<'_, R>,
    transposed: (bool, bool),
    selection: &Selection<A::SelectionArgs>,
) -> Result<(), MatmulSetupError> {
    let rank = lhs.strides.len();
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

    let line_sizes = MatmulLineSizes {
        lhs: 1,
        rhs: 1,
        out: try_tensor_line_size_parallel(
            R::line_size_elem(&eo_elem),
            out.shape,
            out.strides,
            rank - 1,
        )?,
    };

    let batch_lhs: usize = lhs.shape[..lhs.shape.len() - 2].iter().product();
    let batch_rhs: usize = rhs.shape[..rhs.shape.len() - 2].iter().product();

    let problem = MatmulProblem {
        m: m as usize,
        n: n as usize,
        k: k as usize,
        batches: ([batch_lhs].to_vec(), [batch_rhs].to_vec()),
        lhs_layout,
        rhs_layout,
    };

    let plane_size = client.properties().hardware.plane_size_max;

    let plane_dim = match plane_size {
        32 | 64 => plane_size,
        _ => {
            return Err(MatmulSetupError::Unavailable(
                MatmulAvailabilityError::PlaneDimUnsupported {
                    plane_dim: plane_size,
                },
            ));
        }
    };

    if TypeId::of::<MP::ES>() == TypeId::of::<f32>() && tf32::is_supported(client) {
        launch_kernel_concrete::<(ReplaceES<MP, tf32>, TensorMapArgs), R, A>(
            client, lhs, lhs_scale, rhs, rhs_scale, out, problem, line_sizes, plane_dim, selection,
        )
    } else {
        launch_kernel_concrete::<(MP, TensorMapArgs), R, A>(
            client, lhs, lhs_scale, rhs, rhs_scale, out, problem, line_sizes, plane_dim, selection,
        )
    }
}

#[allow(clippy::too_many_arguments, clippy::result_large_err)]
pub fn launch_with_config<'a, MS: MatmulSpec, R: Runtime, A: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    cube_dim: CubeDim,
    cube_count: CubeCount,
    input: InputRuntimeArg<'a, MS, R>,
    output: OutputRuntimeArg<'a, MS, R>,
    cube_count_args: CubeCountInputArgs<'a, R>,
    config: <A::BatchMatmul as BatchMatmulFamily>::Config,
) -> Result<(), MatmulSetupError> {
    unsafe {
        A::BatchMatmul::launch_unchecked::<MS, R>(
            client,
            cube_dim,
            cube_count,
            input,
            output,
            cube_count_args,
            config,
        );
    };

    Ok(())
}
