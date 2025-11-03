use crate::components::{
    AccG, AccS,
    batch::{BatchMatmulFamily, CubeCountInputArgs},
    global::args::{TensorArgs, TensorMapArgs},
};
use crate::components::{
    AvailableLineSizes, InputRuntimeArg, LhsG, LhsS, MatmulAvailabilityError, MatmulLineSizes,
    MatmulProblem, MatmulSelection, MatmulSetupError, MatmulSpec, MatrixLayout, OutputRuntimeArg,
    RhsG, RhsS,
};
use crate::components::{
    InputArg, OutputArg,
    global::args::{ConcreteInputsFactory, ConcreteOutputFactory},
};
use crate::components::{MatmulPrecision, tile::TileMatmulFamily};
use crate::kernels::layered::selector::launch_kernel_concrete;
use crate::{MatmulInputHandle, MatmulInputHandleRef};
use core::any::TypeId;
use cubecl_core::prelude::*;
use cubecl_core::{Runtime, client::ComputeClient, frontend::TensorHandleRef};
use cubecl_runtime::TypeUsage;
use cubecl_std::tensor::{MatrixBatchLayout, TensorHandle, matrix_batch_layout};

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
    client: &ComputeClient<R::Server>,
    lhs: MatmulInputHandle<R, LhsG<MP>>,
    rhs: MatmulInputHandle<R, RhsG<MP>>,
    out: TensorHandle<R, AccG<MP>>,
    selection: &Selection<A::SelectionArgs>,
) -> Result<TensorHandle<R, AccG<MP>>, MatmulSetupError> {
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
    client: &ComputeClient<R::Server>,
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
        lhs_owned = lhs.into_contiguous::<LhsG<MP>>(client);
        &lhs_owned.as_ref()
    } else {
        lhs
    };
    let rhs = if rhs_make_contiguous {
        rhs_owned = rhs.into_contiguous::<RhsG<MP>>(client);
        &rhs_owned.as_ref()
    } else {
        rhs
    };

    launch_inner_ref::<R, (MP, TensorArgs), A>(
        client,
        lhs,
        rhs,
        out,
        (lhs_transposed, rhs_transposed),
        selection,
    )
}

/// Launch a matrix multiplication kernel, with TMA restrictions enabled.
/// TMA doesn't support permuted batches, so checks are slightly different.
///
/// Cmma will be used if available and enabled,
/// otherwise it will fall back on a non-cmma implementation
#[allow(clippy::result_large_err)]
pub fn launch_ref_tma<R: Runtime, MP: MatmulPrecision, A: Algorithm>(
    client: &ComputeClient<R::Server>,
    lhs: &MatmulInputHandleRef<'_, R>,
    rhs: &MatmulInputHandleRef<'_, R>,
    out: &TensorHandleRef<'_, R>,
    selection: &Selection<A::SelectionArgs>,
) -> Result<(), MatmulSetupError> {
    let check_layout = |tensor: &TensorHandleRef<'_, R>| match matrix_batch_layout(tensor.strides) {
        MatrixBatchLayout::Contiguous => (false, false),
        MatrixBatchLayout::MildlyPermuted {
            transposed,
            batch_swap: false,
        } => (false, transposed),
        _ => (true, false),
    };

    let (lhs_make_contiguous, lhs_transposed) = check_layout(lhs.data());
    let (rhs_make_contiguous, rhs_transposed) = check_layout(rhs.data());

    let lhs_owned;
    let rhs_owned;
    let lhs = if lhs_make_contiguous {
        lhs_owned = lhs.into_contiguous::<LhsG<MP>>(client);
        &lhs_owned.as_ref()
    } else {
        lhs
    };
    let rhs = if rhs_make_contiguous {
        rhs_owned = rhs.into_contiguous::<RhsG<MP>>(client);
        &rhs_owned.as_ref()
    } else {
        rhs
    };

    launch_inner_ref::<R, (MP, TensorMapArgs), A>(
        client,
        lhs,
        rhs,
        out,
        (lhs_transposed, rhs_transposed),
        selection,
    )
}

#[allow(clippy::result_large_err, clippy::too_many_arguments)]
fn launch_inner_ref<R: Runtime, MS: MatmulSpec, A: Algorithm>(
    client: &ComputeClient<R::Server>,
    lhs_handle: &MatmulInputHandleRef<'_, R>,
    rhs_handle: &MatmulInputHandleRef<'_, R>,
    out: &TensorHandleRef<'_, R>,
    transposed: (bool, bool),
    selection: &Selection<A::SelectionArgs>,
) -> Result<(), MatmulSetupError>
where
    InputArg<MS>: ConcreteInputsFactory,
    OutputArg<MS>: ConcreteOutputFactory,
{
    let lhs_shape = lhs_handle.shape();
    let rhs_shape = rhs_handle.shape();

    let rank = lhs_shape.len();
    let lhs_elem = LhsG::<MS>::as_type_native().expect("To be a native type");
    let rhs_elem = RhsG::<MS>::as_type_native().expect("To be a native type");
    let acc_elem = AccG::<MS>::as_type_native().expect("To be a native type");

    if !LhsG::<MS>::supported_uses(client).contains(TypeUsage::Conversion)
        || !RhsG::<MS>::supported_uses(client).contains(TypeUsage::Conversion)
        || !AccG::<MS>::supported_uses(client).contains(TypeUsage::Conversion)
    {
        return Err(MatmulSetupError::Unavailable(
            MatmulAvailabilityError::TypesUnavailable {
                lhs: lhs_elem,
                rhs: rhs_elem,
                output: acc_elem,
            },
        ));
    }

    let m = lhs_shape[rank - 2] as u32;
    let k = lhs_shape[rank - 1] as u32;
    let n = rhs_shape[rank - 1] as u32;

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
        lhs_batches: lhs_shape[..lhs_shape.len() - 2].to_vec(),
        rhs_batches: rhs_shape[..rhs_shape.len() - 2].to_vec(),
        out_batches: out.shape[..out.shape.len() - 2].to_vec(),
        lhs_layout,
        rhs_layout,
    };

    let lhs = lhs_handle.data();
    let rhs = rhs_handle.data();

    let line_sizes =
        AvailableLineSizes::from_type_sizes::<R>(lhs.elem_size, rhs.elem_size, out.elem_size);
    let line_sizes = A::filter_line_sizes(line_sizes);
    let mut line_sizes = line_sizes
        .filter_lhs_with_tensor(lhs.strides, lhs.shape, problem.lhs_layout)
        .filter_rhs_with_tensor(rhs.strides, rhs.shape, problem.rhs_layout)
        .filter_out_with_tensor(out.strides, out.shape)
        .pick_max()?;

    // The large line size resulting from dequantizing ends up slower due to restrictions on
    // algorithms. Use this as a quick and dirty fix.
    if lhs_handle.scale().is_some() {
        line_sizes.lhs = 1;
    }
    if rhs_handle.scale().is_some() {
        line_sizes.rhs = 1;
    }

    let fix_plane_dim = |plane_dim: u32| {
        // Sometimes the GPU doesn't support plane instructions and doesn't report the
        // plane size, but we can still execute algorithms that don't use plane instructions.
        //
        // In this case, we set a plane size for the selector to work, defaulting to 32 as it
        // is a common plane size.
        if plane_dim == 0 { 32 } else { plane_dim }
    };

    let plane_dim = fix_plane_dim(A::select_plane_dim::<R>(client));

    launch_inner_ref_fix_dtype::<R, MS, A>(
        client, lhs_handle, rhs_handle, out, problem, line_sizes, plane_dim, selection,
    )
}

#[allow(clippy::result_large_err, clippy::too_many_arguments)]
fn launch_inner_ref_fix_dtype<R: Runtime, MS: MatmulSpec, A: Algorithm>(
    client: &ComputeClient<R::Server>,
    lhs: &MatmulInputHandleRef<'_, R>,
    rhs: &MatmulInputHandleRef<'_, R>,
    out: &TensorHandleRef<'_, R>,
    problem: MatmulProblem,
    line_sizes: MatmulLineSizes,
    plane_dim: u32,
    selection: &Selection<A::SelectionArgs>,
) -> Result<(), MatmulSetupError>
where
    InputArg<MS>: ConcreteInputsFactory,
    OutputArg<MS>: ConcreteOutputFactory,
{
    if <A::TileMatmul as TileMatmulFamily>::requires_accelerator()
        && tf32::supported_uses(client).contains(TypeUsage::Conversion)
    {
        match (
            TypeId::of::<LhsG<MS>>() == TypeId::of::<f32>(),
            TypeId::of::<RhsG<MS>>() == TypeId::of::<f32>(),
        ) {
            (true, true) => launch_kernel_concrete::<
                (
                    (LhsG<MS>, RhsG<MS>, AccG<MS>, tf32, tf32, AccS<MS>),
                    MS::Args,
                ),
                R,
                A,
            >(
                client, lhs, rhs, out, problem, line_sizes, plane_dim, selection,
            ),
            (true, false) => launch_kernel_concrete::<
                (
                    (LhsG<MS>, RhsG<MS>, AccG<MS>, tf32, RhsS<MS>, AccS<MS>),
                    MS::Args,
                ),
                R,
                A,
            >(
                client, lhs, rhs, out, problem, line_sizes, plane_dim, selection,
            ),
            (false, true) => launch_kernel_concrete::<
                (
                    (LhsG<MS>, RhsG<MS>, AccG<MS>, LhsS<MS>, tf32, AccS<MS>),
                    MS::Args,
                ),
                R,
                A,
            >(
                client, lhs, rhs, out, problem, line_sizes, plane_dim, selection,
            ),
            (false, false) => launch_kernel_concrete::<MS, R, A>(
                client, lhs, rhs, out, problem, line_sizes, plane_dim, selection,
            ),
        }
    } else {
        launch_kernel_concrete::<MS, R, A>(
            client, lhs, rhs, out, problem, line_sizes, plane_dim, selection,
        )
    }
}

#[allow(clippy::too_many_arguments, clippy::result_large_err)]
pub fn launch_with_config<'a, MS: MatmulSpec, R: Runtime, A: Algorithm>(
    client: &ComputeClient<R::Server>,
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
