use crate::components::batch::BatchConfig;
use crate::components::global::args::MatmulArgs;
use crate::components::{
    InputArg, InputRuntimeArg, MatmulElems, MatmulLineSizes, MatmulSetupError, OutputRuntimeArg,
};
use crate::components::{
    MatmulProblem, OutputArg,
    global::args::{ConcreteInputsFactory, ConcreteOutputFactory},
};
use crate::kernels::layered::base::Selection;
use crate::kernels::layered::{Algorithm, launch_with_config};
use crate::{MatmulInputHandleRef, components::tile::TileMatmulFamily};
use cubecl_core::prelude::TensorHandleRef;
use cubecl_core::{Runtime, client::ComputeClient};

/// Select which kernel to launch for the given Algorithm.
///
/// Only works for concrete tensor inputs and output.
#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch_kernel_concrete<MA: MatmulArgs, R: Runtime, A: Algorithm>(
    client: &ComputeClient<R>,
    lhs: &MatmulInputHandleRef<'_, R>,
    rhs: &MatmulInputHandleRef<'_, R>,
    out: &TensorHandleRef<'_, R>,
    problem: MatmulProblem,
    line_sizes: MatmulLineSizes,
    plane_dim: u32,
    selection: &Selection<A::SelectionArgs>,
    dtypes: &mut MatmulElems,
) -> Result<(), MatmulSetupError>
where
    InputArg<MA>: ConcreteInputsFactory,
    OutputArg<MA>: ConcreteOutputFactory,
{
    let mut view_line_sizes = line_sizes;

    if let MatmulInputHandleRef::Quantized { scheme, .. } = lhs {
        view_line_sizes.lhs *= scheme.num_quants() as u8;
    }
    if let MatmulInputHandleRef::Quantized { scheme, .. } = rhs {
        view_line_sizes.rhs *= scheme.num_quants() as u8;
    }

    // Prefer output type for stage because it's the same size at best, but often smaller.
    // Having stage == global also enables things like TMA, and an f16 stage for output enables
    // using `stmatrix` on the registers after casting.
    if A::TileMatmul::can_cast_stage_element() {
        dtypes.lhs_stage.dtype = dtypes.lhs_global.dtype;
        dtypes.rhs_stage.dtype = dtypes.rhs_global.dtype;
        dtypes.acc_stage.dtype = dtypes.acc_global.dtype;
    }

    let selection = match selection {
        Selection::Forced(selection) => selection.clone(),
        Selection::Inferred(args) => {
            A::selection(client, &problem, plane_dim, &view_line_sizes, args, dtypes)?
        }
    };
    let config = A::setup(client, &problem, &selection, &view_line_sizes, dtypes)?;
    let cube_count_plan = config.hypercube_config().cube_count_plan(
        &problem,
        client.properties().hardware.max_cube_count.clone(),
    );

    launch_with_config::<MA, R, A>(
        client,
        config.cube_dim(),
        cube_count_plan.resolve(),
        <InputArg<MA> as ConcreteInputsFactory>::create(
            client,
            lhs,
            rhs,
            &selection,
            &problem,
            &line_sizes,
            config,
            dtypes,
        ),
        <OutputArg<MA> as ConcreteOutputFactory>::create(
            client,
            out,
            &selection,
            &problem,
            &line_sizes,
            config,
            dtypes,
        ),
        cube_count_plan.as_args(),
        config,
        dtypes,
    )
}

/// Select which kernel to launch for the given Algorithm.
#[allow(clippy::too_many_arguments)]
pub fn launch_kernel_virtual<'a, MA: MatmulArgs, R: Runtime, A: Algorithm>(
    client: &ComputeClient<R>,
    input: InputRuntimeArg<'a, MA, R>,
    output: OutputRuntimeArg<'a, MA, R>,
    problem: MatmulProblem,
    view_line_sizes: MatmulLineSizes,
    plane_dim: u32,
    selection: &Selection<A::SelectionArgs>,
    dtypes: &mut MatmulElems,
) -> Result<(), MatmulSetupError> {
    // Prefer output type for stage because it's the same size at best, but often smaller.
    // Having stage == global also enables things like TMA, and an f16 stage for output enables
    // using `stmatrix` on the registers after casting.
    if A::TileMatmul::can_cast_stage_element() {
        dtypes.lhs_stage.dtype = dtypes.lhs_global.dtype;
        dtypes.rhs_stage.dtype = dtypes.rhs_global.dtype;
        dtypes.acc_stage.dtype = dtypes.acc_global.dtype;
    }

    let selection = match selection {
        Selection::Forced(selection) => selection.clone(),
        Selection::Inferred(args) => {
            A::selection(client, &problem, plane_dim, &view_line_sizes, args, dtypes)?
        }
    };
    let config = A::setup(client, &problem, &selection, &view_line_sizes, dtypes)?;

    let cube_count_plan = config.hypercube_config().cube_count_plan(
        &problem,
        client.properties().hardware.max_cube_count.clone(),
    );

    launch_with_config::<MA, R, A>(
        client,
        config.cube_dim(),
        cube_count_plan.resolve(),
        input,
        output,
        cube_count_plan.as_args(),
        config,
        dtypes,
    )
}
