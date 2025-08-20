use cubecl_core::prelude::TensorHandleRef;
use cubecl_core::{Runtime, client::ComputeClient};
use cubecl_matmul::{
    MatmulInputHandleRef,
    components::{
        InputArg, InputRuntimeArg, MatmulLineSizes, MatmulSelection, MatmulSpec, OutputArg,
        OutputRuntimeArg,
        global::{GlobalConfig as _, args::ConcreteOutputFactory},
    },
};

use crate::{
    components::{
        ConvSetupError, ConvolutionProblem,
        global::{args::ConcreteInputsFactory, entry_point::ConvolutionLaunch},
    },
    kernels::layered::algorithm::Algorithm,
};

/// Select which kernel to launch for the given Algorithm.
///
/// Only works for concrete tensor inputs and output.
#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch_kernel_concrete<MS: MatmulSpec, R: Runtime, A: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: &MatmulInputHandleRef<'_, R>,
    weight: &MatmulInputHandleRef<'_, R>,
    bias: &Option<TensorHandleRef<'_, R>>,
    out: &TensorHandleRef<'_, R>,
    problem: ConvolutionProblem,
    line_sizes: MatmulLineSizes,
    selection: MatmulSelection,
) -> Result<(), ConvSetupError>
where
    InputArg<MS>: ConcreteInputsFactory,
    OutputArg<MS>: ConcreteOutputFactory,
{
    let config = A::setup::<R, MS::Precision>(client, &problem, &selection, &line_sizes)?;

    let input = <InputArg<MS> as ConcreteInputsFactory>::create(
        input,
        weight,
        bias.as_ref(),
        &selection,
        &problem,
        &line_sizes,
    );
    let output = <OutputArg<MS> as ConcreteOutputFactory>::create(
        out,
        &selection,
        &problem.as_matmul_problem(),
        &line_sizes,
    );

    unsafe {
        A::GlobalConvolution::launch_unchecked::<MS, R>(
            client,
            config.cube_dim(),
            A::cube_count(&selection, &problem),
            input,
            output,
            &problem,
            config,
        );
    }

    Ok(())
}

/// Select which kernel to launch for the given Algorithm.
pub fn launch_kernel_virtual<'a, MS: MatmulSpec, R: Runtime, A: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: InputRuntimeArg<'a, MS, R>,
    output: OutputRuntimeArg<'a, MS, R>,
    problem: ConvolutionProblem,
    line_sizes: MatmulLineSizes,
    selection: MatmulSelection,
) -> Result<(), ConvSetupError> {
    let config = A::setup::<R, MS::Precision>(client, &problem, &selection, &line_sizes)?;

    unsafe {
        A::GlobalConvolution::launch_unchecked::<MS, R>(
            client,
            config.cube_dim(),
            A::cube_count(&selection, &problem),
            input,
            output,
            &problem,
            config,
        );
    }

    Ok(())
}
