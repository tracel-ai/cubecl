use crate::components::batch::BatchConfig;
use crate::components::{InputRuntimeArg, MatmulLineSizes, MatmulPrecision, OutputRuntimeArg};
use crate::kernels::matmul::base::Selection;
use crate::kernels::matmul::{Algorithm, launch_with_config};
use crate::{
    components::{
        InputArg, MatmulProblem, MatmulSpec, OutputArg,
        global::args::{ConcreteInputsFactory, ConcreteOutputFactory},
    },
    kernels::MatmulSetupError,
};
use cubecl_core::frontend::CubePrimitive;
use cubecl_core::prelude::TensorHandleRef;
use cubecl_core::{Runtime, client::ComputeClient};

/// Select which kernel to launch for the given Algorithm.
///
/// Only works for concrete tensor inputs and output.
#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch_kernel_concrete<MS: MatmulSpec, R: Runtime, A: Algorithm>(
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
) -> Result<(), MatmulSetupError>
where
    InputArg<MS>: ConcreteInputsFactory,
    OutputArg<MS>: ConcreteOutputFactory,
{
    let elem_stage = <MS::Precision as MatmulPrecision>::ES::as_elem_native_unchecked();
    let elem_acc = <MS::Precision as MatmulPrecision>::EA::as_elem_native_unchecked();

    let selection = match selection {
        Selection::Forced(selection) => selection.clone(),
        Selection::Inferred(args) => {
            A::selection::<R>(client, &problem, plane_dim, elem_stage, elem_acc, args)
        }
    };
    println!("{selection:?}");
    let config = A::setup::<MS::Precision, R>(client, &problem, &selection, &line_sizes)?;

    let line_sizes = config.line_sizes();

    launch_with_config::<MS, R, A>(
        client,
        config.cube_dim(),
        config.cube_count(&problem),
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
pub fn launch_kernel_virtual<'a, MS: MatmulSpec, R: Runtime, A: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: InputRuntimeArg<'a, MS, R>,
    output: OutputRuntimeArg<'a, MS, R>,
    problem: MatmulProblem,
    line_sizes: MatmulLineSizes,
    plane_dim: u32,
    selection: &Selection<A::SelectionArgs>,
) -> Result<(), MatmulSetupError> {
    let elem_stage = <MS::Precision as MatmulPrecision>::ES::as_elem_native_unchecked();
    let elem_acc = <MS::Precision as MatmulPrecision>::EA::as_elem_native_unchecked();

    let selection = match selection {
        Selection::Forced(selection) => selection.clone(),
        Selection::Inferred(args) => {
            A::selection::<R>(client, &problem, plane_dim, elem_stage, elem_acc, args)
        }
    };
    let config = A::setup::<MS::Precision, R>(client, &problem, &selection, &line_sizes)?;

    launch_with_config::<MS, R, A>(
        client,
        config.cube_dim(),
        config.cube_count(&problem),
        input,
        output,
        config,
    )
}
