use crate::components::batch::{BatchMatmulFamily, CubeCountInputArgs};
use crate::components::global::args::TensorArgs;
use crate::components::{
    AvailableLineSizes, InputRuntimeArg, LhsG, LhsS, MatmulAvailabilityError, MatmulLineSizes,
    MatmulPrecision, MatmulProblem, MatmulSelection, MatmulSetupError, MatmulSpec, MatrixLayout,
    OutputRuntimeArg, RhsG, RhsS,
};
use crate::components::{global::args::TensorMapArgs, tile::TileMatmulFamily};
use crate::kernels::layered::selector::launch_kernel_concrete;
use crate::{MatmulInputHandle, MatmulInputHandleRef};
use core::any::TypeId;
use cubecl_core::{Feature, prelude::*, try_tensor_line_size_parallel};
use cubecl_core::{Runtime, client::ComputeClient, frontend::TensorHandleRef};
use cubecl_std::tensor::{
    MatrixBatchLayout, TensorHandle, into_contiguous_pitched, matrix_batch_layout,
};

use super::Algorithm;

#[derive(Debug, Clone)]
pub enum Selection<S> {
    /// Use a predefined MatmulSelection
    Forced(MatmulSelection),
    /// Allows to give limited MatmulSelection information, and the rest is inferred from it
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
    lhs: MatmulInputHandle<R, LhsG<MP>>,
    rhs: MatmulInputHandle<R, RhsG<MP>>,
    out: TensorHandle<R, MP::EO>,
    selection: &Selection<A::SelectionArgs>,
) -> Result<TensorHandle<R, MP::EO>, MatmulSetupError> {
    let result = launch_ref::<R, MP, A>(
        client,
        &lhs.as_ref(),
        &rhs.as_ref(),
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
    lhs: &MatmulInputHandleRef<'_, R>,
    rhs: &MatmulInputHandleRef<'_, R>,
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

    let (lhs_make_contiguous, lhs_transposed) = check_layout(lhs.data());
    let (rhs_make_contiguous, rhs_transposed) = check_layout(rhs.data());

    let lhs_owned;
    let rhs_owned;
    let lhs = if lhs_make_contiguous {
        lhs_owned = match lhs {
            MatmulInputHandleRef::Normal(data) => {
                MatmulInputHandle::Normal(into_contiguous_pitched::<R, LhsG<MP>>(client, data))
            }
            MatmulInputHandleRef::Quantized { .. } => unimplemented!(),
        };
        &lhs_owned.as_ref()
    } else {
        lhs
    };
    let rhs = if rhs_make_contiguous {
        rhs_owned = match rhs {
            MatmulInputHandleRef::Normal(data) => {
                MatmulInputHandle::Normal(into_contiguous_pitched::<R, RhsG<MP>>(client, data))
            }
            MatmulInputHandleRef::Quantized { .. } => unimplemented!(),
        };
        &rhs_owned.as_ref()
    } else {
        rhs
    };

    launch_inner_ref::<R, MP, A>(
        client,
        lhs,
        rhs,
        out,
        (lhs_transposed, rhs_transposed),
        selection,
    )
}

#[allow(clippy::result_large_err, clippy::too_many_arguments)]
fn launch_inner_ref<R: Runtime, MP: MatmulPrecision, A: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs_handle: &MatmulInputHandleRef<'_, R>,
    rhs_handle: &MatmulInputHandleRef<'_, R>,
    out: &TensorHandleRef<'_, R>,
    transposed: (bool, bool),
    selection: &Selection<A::SelectionArgs>,
) -> Result<(), MatmulSetupError> {
    let lhs = lhs_handle.data();
    let rhs = rhs_handle.data();

    let rank = lhs.strides.len();
    let lhs_elem = LhsG::<MP>::as_elem_native().expect("To be a native type");
    let rhs_elem = RhsG::<MP>::as_elem_native().expect("To be a native type");
    let eo_elem = MP::EO::as_elem_native().expect("To be a native type");

    if !client.properties().feature_enabled(Feature::Type(lhs_elem))
        || !client.properties().feature_enabled(Feature::Type(eo_elem))
    {
        return Err(MatmulSetupError::Unavailable(
            MatmulAvailabilityError::TypesUnavailable {
                lhs: lhs_elem,
                rhs: rhs_elem,
                output: eo_elem,
            },
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
        lhs_batches: lhs.shape[..lhs.shape.len() - 2].to_vec(),
        rhs_batches: rhs.shape[..rhs.shape.len() - 2].to_vec(),
        lhs_layout,
        rhs_layout,
    };

    let line_sizes = AvailableLineSizes::from_elem_types::<R>(&lhs_elem, &rhs_elem, &eo_elem);
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
        client, lhs_handle, rhs_handle, out, problem, line_sizes, plane_dim, selection,
    )
}

#[allow(clippy::result_large_err, clippy::too_many_arguments)]
fn launch_inner_ref_fix_dtype<R: Runtime, MP: MatmulPrecision, A: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: &MatmulInputHandleRef<'_, R>,
    rhs: &MatmulInputHandleRef<'_, R>,
    out: &TensorHandleRef<'_, R>,
    problem: MatmulProblem,
    line_sizes: MatmulLineSizes,
    plane_dim: u32,
    selection: &Selection<A::SelectionArgs>,
) -> Result<(), MatmulSetupError> {
    if <A::TileMatmul as TileMatmulFamily>::requires_accelerator() && tf32::is_supported(client) {
        match (
            TypeId::of::<LhsG<MP>>() == TypeId::of::<f32>(),
            TypeId::of::<RhsG<MP>>() == TypeId::of::<f32>(),
        ) {
            (true, true) => {
                launch_kernel_concrete::<((f32, f32, tf32, tf32, MP::EA, MP::EO), TensorArgs), R, A>(
                    client, lhs, rhs, out, problem, line_sizes, plane_dim, selection,
                )
            }
            (true, false) => launch_kernel_concrete::<
                ((f32, RhsG<MP>, tf32, RhsS<MP>, MP::EA, MP::EO), TensorArgs),
                R,
                A,
            >(
                client, lhs, rhs, out, problem, line_sizes, plane_dim, selection,
            ),
            (false, true) => launch_kernel_concrete::<
                ((LhsG<MP>, f32, LhsS<MP>, tf32, MP::EA, MP::EO), TensorArgs),
                R,
                A,
            >(
                client, lhs, rhs, out, problem, line_sizes, plane_dim, selection,
            ),
            (false, false) => launch_kernel_concrete::<(MP, TensorArgs), R, A>(
                client, lhs, rhs, out, problem, line_sizes, plane_dim, selection,
            ),
        }
    } else {
        launch_kernel_concrete::<(MP, TensorArgs), R, A>(
            client, lhs, rhs, out, problem, line_sizes, plane_dim, selection,
        )
    }
}

#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn matmul_cmma_tma_ref_no_check<R: Runtime, MP: MatmulPrecision, A: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs_handle: &MatmulInputHandleRef<'_, R>,
    rhs_handle: &MatmulInputHandleRef<'_, R>,
    out: &TensorHandleRef<'_, R>,
    transposed: (bool, bool),
    selection: &Selection<A::SelectionArgs>,
) -> Result<(), MatmulSetupError> {
    let lhs = lhs_handle.data();
    let rhs = rhs_handle.data();

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
        lhs_batches: [batch_lhs].to_vec(),
        rhs_batches: [batch_rhs].to_vec(),
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

    if tf32::is_supported(client) {
        match (
            TypeId::of::<LhsG<MP>>() == TypeId::of::<f32>(),
            TypeId::of::<RhsG<MP>>() == TypeId::of::<f32>(),
        ) {
            (true, true) => launch_kernel_concrete::<
                ((f32, f32, tf32, tf32, MP::EA, MP::EO), TensorMapArgs),
                R,
                A,
            >(
                client, lhs_handle, rhs_handle, out, problem, line_sizes, plane_dim, selection,
            ),
            (true, false) => launch_kernel_concrete::<
                (
                    (f32, RhsG<MP>, tf32, RhsS<MP>, MP::EA, MP::EO),
                    TensorMapArgs,
                ),
                R,
                A,
            >(
                client, lhs_handle, rhs_handle, out, problem, line_sizes, plane_dim, selection,
            ),
            (false, true) => launch_kernel_concrete::<
                (
                    (LhsG<MP>, f32, LhsS<MP>, tf32, MP::EA, MP::EO),
                    TensorMapArgs,
                ),
                R,
                A,
            >(
                client, lhs_handle, rhs_handle, out, problem, line_sizes, plane_dim, selection,
            ),
            (false, false) => launch_kernel_concrete::<(MP, TensorMapArgs), R, A>(
                client, lhs_handle, rhs_handle, out, problem, line_sizes, plane_dim, selection,
            ),
        }
    } else {
        launch_kernel_concrete::<(MP, TensorMapArgs), R, A>(
            client, lhs_handle, rhs_handle, out, problem, line_sizes, plane_dim, selection,
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
    cube_count_input: CubeCountInputArgs<'a, R>,
    config: <A::BatchMatmul as BatchMatmulFamily>::Config,
) -> Result<(), MatmulSetupError> {
    unsafe {
        A::BatchMatmul::launch_unchecked::<MS, R>(
            client,
            cube_dim,
            cube_count,
            input,
            output,
            cube_count_input,
            config,
        );
    };

    Ok(())
}
