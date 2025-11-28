use crate::components::ConvGemmConfig as _;
use cubecl_core::prelude::TensorHandleRef;
use cubecl_core::{Runtime, client::ComputeClient};
use cubecl_matmul::components::MatmulElems;
use cubecl_matmul::components::global::args::MatmulArgs;
use cubecl_matmul::{
    MatmulInputHandleRef,
    components::{
        InputArg, InputRuntimeArg, MatmulLineSizes, MatmulSelection, OutputArg, OutputRuntimeArg,
    },
};

use crate::{
    components::{
        ConvSetupError, ConvolutionProblem,
        global::{
            args::{ConcreteInputsFactory, ConcreteOutputFactory},
            entry_point::ConvolutionLaunch,
        },
    },
    kernels::layered::algorithm::Algorithm,
};

/// Select which kernel to launch for the given Algorithm.
///
/// Only works for concrete tensor inputs and output.
#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch_kernel_concrete<R: Runtime, A: Algorithm>(
    client: &ComputeClient<R>,
    input: &MatmulInputHandleRef<'_, R>,
    weight: &MatmulInputHandleRef<'_, R>,
    bias: &Option<TensorHandleRef<'_, R>>,
    out: &TensorHandleRef<'_, R>,
    problem: ConvolutionProblem,
    line_sizes: MatmulLineSizes,
    selection: MatmulSelection,
    dtypes: &MatmulElems,
) -> Result<(), ConvSetupError>
where
    InputArg<A::Args>: ConcreteInputsFactory,
    OutputArg<A::Args>: ConcreteOutputFactory,
{
    let config = A::setup(client, &problem, &selection, &line_sizes, dtypes)?;

    let input = <InputArg<A::Args> as ConcreteInputsFactory>::create(
        client,
        input,
        weight,
        bias.as_ref(),
        &selection,
        &problem,
        &line_sizes,
        config,
        dtypes,
    );
    let output = <OutputArg<A::Args> as ConcreteOutputFactory>::create(
        client,
        out,
        &selection,
        &problem,
        &line_sizes,
        config,
        dtypes,
    );

    let result = unsafe {
        A::GlobalConvolution::launch_unchecked::<A::Args, R>(
            client,
            config.cube_dim(),
            A::cube_count(&selection, &problem),
            input,
            output,
            &problem,
            config,
            dtypes,
        )
    };

    match result {
        Ok(_) => Ok(()),
        Err(err) => Err(ConvSetupError::Launch(err)),
    }
}

/// Select which kernel to launch for the given Algorithm.
pub fn launch_kernel_virtual<'a, MA: MatmulArgs, R: Runtime, A: Algorithm>(
    client: &ComputeClient<R>,
    input: InputRuntimeArg<'a, MA, R>,
    output: OutputRuntimeArg<'a, MA, R>,
    problem: ConvolutionProblem,
    line_sizes: MatmulLineSizes,
    selection: MatmulSelection,
    dtypes: &MatmulElems,
) -> Result<(), ConvSetupError> {
    let config = A::setup(client, &problem, &selection, &line_sizes, dtypes)?;

    let result = unsafe {
        A::GlobalConvolution::launch_unchecked::<MA, R>(
            client,
            config.cube_dim(),
            A::cube_count(&selection, &problem),
            input,
            output,
            &problem,
            config,
            dtypes,
        )
    };

    match result {
        Ok(_) => Ok(()),
        Err(err) => Err(ConvSetupError::Launch(err)),
    }
}
