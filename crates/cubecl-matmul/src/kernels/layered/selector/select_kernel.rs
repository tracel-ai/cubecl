use crate::MatmulInputHandleRef;
use crate::components::batch::BatchConfig;
use crate::components::{
    InputArg, InputRuntimeArg, MatmulElems, MatmulLineSizes, MatmulSetupError, OutputRuntimeArg,
};
use crate::components::{
    MatmulProblem, MatmulSpec, OutputArg,
    global::args::{ConcreteInputsFactory, ConcreteOutputFactory},
};
use crate::kernels::layered::base::Selection;
use crate::kernels::layered::{Algorithm, launch_with_config};
use cubecl_core::prelude::TensorHandleRef;
use cubecl_core::{Runtime, client::ComputeClient};

/// Select which kernel to launch for the given Algorithm.
///
/// Only works for concrete tensor inputs and output.
#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch_kernel_concrete<MS: MatmulSpec, R: Runtime, A: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
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
    let elems = MatmulElems::new::<MS::Precision>();

    let selection = match selection {
        Selection::Forced(selection) => selection.clone(),
        Selection::Inferred(args) => A::selection::<R>(client, &problem, plane_dim, elems, args)?,
    };
    let config = A::setup::<MS::Precision, R>(client, &problem, &selection, &line_sizes)?;
    let cube_count_plan = config.hypercube_config().cube_count_plan(
        &problem,
        client.properties().hardware.max_cube_count.clone(),
    );

    let line_sizes = config.line_sizes();

    launch_with_config::<MS, R, A>(
        client,
        config.cube_dim(),
        cube_count_plan.resolve(),
        <InputArg<MS> as ConcreteInputsFactory>::create(
            lhs,
            rhs,
            &selection,
            &problem,
            &line_sizes,
        ),
        <OutputArg<MS> as ConcreteOutputFactory>::create(out, &selection, &problem, &line_sizes),
        cube_count_plan.as_args(),
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
    let elems = MatmulElems::new::<MS::Precision>();

    let selection = match selection {
        Selection::Forced(selection) => selection.clone(),
        Selection::Inferred(args) => A::selection::<R>(client, &problem, plane_dim, elems, args)?,
    };
    let config = A::setup::<MS::Precision, R>(client, &problem, &selection, &line_sizes)?;

    let cube_count_plan = config.hypercube_config().cube_count_plan(
        &problem,
        client.properties().hardware.max_cube_count.clone(),
    );

    launch_with_config::<MS, R, A>(
        client,
        config.cube_dim(),
        cube_count_plan.resolve(),
        input,
        output,
        cube_count_plan.as_args(),
        config,
    )
}
