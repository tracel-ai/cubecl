use cubecl_core::ir::Elem;
use cubecl_core::prelude::TensorHandleRef;
use cubecl_core::{Runtime, client::ComputeClient};

use crate::components::batch::BatchConfig;
use crate::components::{
    AvailableLineSizes, InputRuntimeArg, MatmulLineSizes, MatmulPrecision, OutputRuntimeArg,
};
use crate::kernels::matmul::Algorithm;
use crate::kernels::matmul::base::launch_matmul;
use crate::{
    components::{
        InputArg, MatmulProblem, MatmulSpec, OutputArg,
        global::args::{ConcreteInputsFactory, ConcreteOutputFactory},
    },
    kernels::MatmulSetupError,
};
use cubecl_core::frontend::CubePrimitive;

/// Select which kernel to launch for the given Algorithm.
///
/// Only works for concrete tensor inputs and output.
#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn select_kernel_concrete<MS: MatmulSpec, R: Runtime, A: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: &TensorHandleRef<'_, R>,
    lhs_scale: &Option<TensorHandleRef<'_, R>>,
    rhs: &TensorHandleRef<'_, R>,
    rhs_scale: &Option<TensorHandleRef<'_, R>>,
    out: &TensorHandleRef<'_, R>,
    problem: MatmulProblem,
    forced_line_sizes: Option<MatmulLineSizes>,
    plane_dim: u32,
) -> Result<(), MatmulSetupError>
where
    InputArg<MS>: ConcreteInputsFactory,
    OutputArg<MS>: ConcreteOutputFactory,
{
    let elem_in = <MS::Precision as MatmulPrecision>::EI::as_elem_native_unchecked();
    let elem_stage = <MS::Precision as MatmulPrecision>::ES::as_elem_native_unchecked();
    let elem_acc = <MS::Precision as MatmulPrecision>::EA::as_elem_native_unchecked();
    let elem_out = <MS::Precision as MatmulPrecision>::EO::as_elem_native_unchecked();

    let selection = A::selection::<R>(client, &problem, plane_dim, elem_stage, elem_acc);
    let available_line_sizes =
        find_available_line_sizes::<MS, R>(forced_line_sizes, &elem_in, &elem_out);
    let config = A::setup::<MS::Precision, R>(client, &problem, &selection, available_line_sizes)?;

    let line_sizes = config.line_sizes();

    launch_matmul::<MS, R, A>(
        client,
        config.cube_dim(),
        A::cube_count(&problem, &config),
        <InputArg<MS> as ConcreteInputsFactory>::create(
            lhs,
            lhs_scale,
            rhs,
            rhs_scale,
            &selection,
            &problem,
            &line_sizes,
        ),
        <OutputArg<MS> as ConcreteOutputFactory>::create(out, &selection, &problem, &line_sizes),
        config,
    )
}

/// Select which kernel to launch for the given Algorithm.
pub fn select_kernel_virtual<'a, MS: MatmulSpec, R: Runtime, A: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: InputRuntimeArg<'a, MS, R>,
    output: OutputRuntimeArg<'a, MS, R>,
    problem: MatmulProblem,
    forced_line_sizes: Option<MatmulLineSizes>,
    plane_dim: u32,
) -> Result<(), MatmulSetupError> {
    let elem_in = <MS::Precision as MatmulPrecision>::EI::as_elem_native_unchecked();
    let elem_stage = <MS::Precision as MatmulPrecision>::ES::as_elem_native_unchecked();
    let elem_acc = <MS::Precision as MatmulPrecision>::EA::as_elem_native_unchecked();
    let elem_out = <MS::Precision as MatmulPrecision>::EO::as_elem_native_unchecked();

    let selection = A::selection::<R>(client, &problem, plane_dim, elem_stage, elem_acc);
    let available_line_sizes =
        find_available_line_sizes::<MS, R>(forced_line_sizes, &elem_in, &elem_out);
    let config = A::setup::<MS::Precision, R>(client, &problem, &selection, available_line_sizes)?;

    launch_matmul::<MS, R, A>(
        client,
        config.cube_dim(),
        A::cube_count(&problem, &config),
        input,
        output,
        config,
    )
}

// TODO
// At the moment, because of fusion, we can force line sizes
// But this may make the matmul kernel setup fail.
// Would be better to add a constraint on fusion
fn find_available_line_sizes<MS: MatmulSpec, R: Runtime>(
    forced_line_sizes: Option<MatmulLineSizes>,
    elem_in: &Elem,
    elem_out: &Elem,
) -> AvailableLineSizes {
    match forced_line_sizes {
        Some(forced_line_sizes) => forced_line_sizes.into(),
        None => AvailableLineSizes::from_elem_types::<R>(elem_in, elem_out),
    }
}
