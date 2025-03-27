use std::marker::PhantomData;

use cubecl_core::prelude::{LaunchArg, Numeric};

use super::global::args::{MatmulArgs, TensorArgs};

/// Matrix multiplication spec definiting each element types used in the computation as well as
/// how the arguments are passed to the kernel.
pub trait MatmulSpec: Send + Sync + Clone + 'static {
    type Precision: MatmulPrecision;
    /// How the input and output tensors are passed as arguments.
    type Args: MatmulArgs;
}

/// Matrix multiplication precisions.
pub trait MatmulPrecision: Send + Sync + Clone + 'static {
    const QUANTIZED: bool;

    /// Element type of each input and output tensors of the kernel.
    type EG: Numeric;
    /// Element type for the shared memories used to read inputs.
    type ES: Numeric;
    /// Element type for the shared memories or fragments used to accumulate
    /// smaller matmul results before writing to the output tensor.
    type EA: Numeric;
}

impl<EG: Numeric, ES: Numeric, EA: Numeric> MatmulPrecision for (EG, ES, EA) {
    const QUANTIZED: bool = false;
    type EG = EG;
    type ES = ES;
    type EA = EA;
}

/// Input argument
pub type InputArg<MS> = <Args<MS> as MatmulArgs>::Input<EG<MS>>;

/// Output argument
pub type OutputArg<MS> = <Args<MS> as MatmulArgs>::Output<EG<MS>>;

/// Input runtime argument
pub type InputRuntimeArg<'a, MS, R> = <InputArg<MS> as LaunchArg>::RuntimeArg<'a, R>;

/// Output runtime argument
pub type OutputRuntimeArg<'a, MS, R> = <OutputArg<MS> as LaunchArg>::RuntimeArg<'a, R>;

pub type EG<MS> = <<MS as MatmulSpec>::Precision as MatmulPrecision>::EG;
pub type ES<MS> = <<MS as MatmulSpec>::Precision as MatmulPrecision>::ES;
pub type EA<MS> = <<MS as MatmulSpec>::Precision as MatmulPrecision>::EA;
pub type Args<MS> = <MS as MatmulSpec>::Args;

/// Specification for a simple standard matmul using global tensor as inputs.
#[derive(Clone)]
pub struct SingleMatmulSpec<EG, ES, EA, Args = TensorArgs> {
    _eg: PhantomData<EG>,
    _es: PhantomData<ES>,
    _ea: PhantomData<EA>,
    _args: PhantomData<Args>,
}

impl<Args: MatmulArgs, EG: Numeric, ES: Numeric, EA: Numeric> MatmulSpec
    for SingleMatmulSpec<EG, ES, EA, Args>
{
    type Precision = (EG, ES, EA);
    type Args = Args;
}
