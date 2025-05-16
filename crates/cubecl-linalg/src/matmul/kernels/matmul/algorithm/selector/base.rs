use cubecl_core::prelude::TensorHandleRef;
use cubecl_core::{Runtime, client::ComputeClient};

use crate::matmul::components::stage::StageVectorization;
use crate::matmul::components::{InputRuntimeArg, MatmulPrecision, MatmulSize, OutputRuntimeArg};
use crate::matmul::kernels::matmul::Algorithm;
use crate::matmul::{
    components::{
        CompleteStageTiling, InputArg, MatmulProblem, MatmulSpec, OutputArg,
        global::args::{ConcreteInputsFactory, ConcreteOutputFactory},
    },
    kernels::{MatmulLaunchError, matmul::base::matmul_cube_preparation},
};
use cubecl_core::frontend::CubePrimitive;

pub trait MatmulSelection {
    fn tile_shape(&self) -> MatmulSize;
    fn tile_count(&self) -> MatmulSize;
}

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
    plane_dim: u32,
) -> Result<(), MatmulLaunchError>
where
    InputArg<MS>: ConcreteInputsFactory,
    OutputArg<MS>: ConcreteOutputFactory,
{
    let selection = A::selection::<R>(
        client,
        &problem,
        plane_dim,
        <MS::Precision as MatmulPrecision>::ES::as_elem_native_unchecked(),
        <MS::Precision as MatmulPrecision>::EA::as_elem_native_unchecked(),
    );

    let config_input = CompleteStageTiling {
        tile_shape: selection.tile_shape(),
        tile_count: selection.tile_count(),
    };

    let vectorization = StageVectorization {
        stage_line_size: 0,
        stage_elem_padding: 0,
    };
    matmul_cube_preparation::<MS, R, A>(
        client,
        <InputArg<MS> as ConcreteInputsFactory>::create(
            lhs, lhs_scale, rhs, rhs_scale, &selection, &problem,
        ),
        <OutputArg<MS> as ConcreteOutputFactory>::create(out, &selection, &problem),
        problem,
        (
            (
                config_input,
                A::stage_buffering_strategy(),
                vectorization,
                A::num_stages(),
            ),
            A::loading_precompute_strategy(),
        ),
        selection,
    )
}

/// Select which kernel to launch for the given Algorithm.
pub fn select_kernel_virtual<'a, MS: MatmulSpec, R: Runtime, A: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: InputRuntimeArg<'a, MS, R>,
    output: OutputRuntimeArg<'a, MS, R>,
    problem: MatmulProblem,
    plane_dim: u32,
) -> Result<(), MatmulLaunchError> {
    let selection = A::selection::<R>(
        client,
        &problem,
        plane_dim,
        <MS::Precision as MatmulPrecision>::ES::as_elem_native_unchecked(),
        <MS::Precision as MatmulPrecision>::EA::as_elem_native_unchecked(),
    );

    let config_input = CompleteStageTiling {
        tile_shape: selection.tile_shape(),
        tile_count: selection.tile_count(),
    };
    let vectorization = StageVectorization {
        stage_line_size: 0,
        stage_elem_padding: 0,
    };
    matmul_cube_preparation::<MS, R, A>(
        client,
        input,
        output,
        problem,
        (
            (
                config_input,
                A::stage_buffering_strategy(),
                vectorization,
                A::num_stages(),
            ),
            A::loading_precompute_strategy(),
        ),
        selection,
    )
}
