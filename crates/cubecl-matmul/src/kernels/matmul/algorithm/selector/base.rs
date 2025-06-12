use cubecl_core::prelude::TensorHandleRef;
use cubecl_core::{Runtime, client::ComputeClient};

use crate::components::global::load::LoaderMode;
use crate::components::stage::{PartitionBuffering, StageVectorization};
use crate::components::{
    InputRuntimeArg, LoadSpecializationConfig, MatmulLineSizes, MatmulPrecision, OutputRuntimeArg,
    TilingScheme,
};
use crate::kernels::matmul::{Algorithm, LoadingPrecomputeStrategy};
use crate::{
    components::{
        InputArg, MatmulProblem, MatmulSpec, OutputArg,
        global::args::{ConcreteInputsFactory, ConcreteOutputFactory},
    },
    kernels::{MatmulSetupError, matmul::base::matmul_cube_preparation},
};
use cubecl_core::frontend::CubePrimitive;

#[derive(Debug)]
pub struct MatmulSelection {
    pub plane_dim: u32,
    pub tiling_scheme: TilingScheme,
    pub stage_vectorization: StageVectorization,
    pub quantized: bool,
    pub partition_buffering: PartitionBuffering,
    pub loading_precompute_strategy: LoadingPrecomputeStrategy,
    pub loader_mode: LoaderMode,
    pub load_specialization_config: LoadSpecializationConfig,
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
    // Option of MAX line sizes, overridable in components
    line_sizes: Option<MatmulLineSizes>,
    plane_dim: u32,
) -> Result<(), MatmulSetupError>
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

    let line_sizes = line_sizes.unwrap_or(A::line_sizes(
        &problem,
        R::line_size_elem(&<MS::Precision as MatmulPrecision>::EI::as_elem_native_unchecked()),
        R::line_size_elem(&<MS::Precision as MatmulPrecision>::EO::as_elem_native_unchecked()),
        &selection,
    ));

    matmul_cube_preparation::<MS, R, A>(
        client,
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
        problem,
        &line_sizes,
        A::global_input(&selection),
        selection,
    )
}

/// Select which kernel to launch for the given Algorithm.
pub fn select_kernel_virtual<'a, MS: MatmulSpec, R: Runtime, A: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: InputRuntimeArg<'a, MS, R>,
    output: OutputRuntimeArg<'a, MS, R>,
    problem: MatmulProblem,
    line_sizes: Option<MatmulLineSizes>,
    plane_dim: u32,
) -> Result<(), MatmulSetupError> {
    let selection = A::selection::<R>(
        client,
        &problem,
        plane_dim,
        <MS::Precision as MatmulPrecision>::ES::as_elem_native_unchecked(),
        <MS::Precision as MatmulPrecision>::EA::as_elem_native_unchecked(),
    );

    let line_sizes = line_sizes.unwrap_or(A::line_sizes(
        &problem,
        R::line_size_elem(&<MS::Precision as MatmulPrecision>::EI::as_elem_native_unchecked()),
        R::line_size_elem(&<MS::Precision as MatmulPrecision>::EO::as_elem_native_unchecked()),
        &selection,
    ));

    matmul_cube_preparation::<MS, R, A>(
        client,
        input,
        output,
        problem,
        &line_sizes,
        A::global_input(&selection),
        selection,
    )
}
