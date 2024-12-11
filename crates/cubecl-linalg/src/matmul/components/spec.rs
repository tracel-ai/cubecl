use std::marker::PhantomData;

use cubecl_core::prelude::{LaunchArg, Numeric};

use super::global::args::{MatmulArgs, TensorArgs};

/// Matrix multiplication spec definiting each element types used in the computation as well as
/// how the arguments are passed to the kernel.
pub trait MatmulSpec: Send + Sync + Clone + 'static {
    /// Element type of each input and output tensor of the kernel.
    type EG: Numeric;
    /// Element type of the intermediate representation of the inputs.
    type ES: Numeric;
    /// Element type of the intermediate representation of the output accumulator.
    type EA: Numeric;
    /// How the input and output tensors are passed as arguments.
    type Args: MatmulArgs<Self::EG>;
}

/// Input argument
pub type InputArg<MS> = <Args<MS> as MatmulArgs<EG<MS>>>::Input;

/// Output argument
pub type OutputArg<MS> = <Args<MS> as MatmulArgs<EG<MS>>>::Output;

/// Input runtime argument
pub type InputRuntimeArg<'a, MS, R> = <InputArg<MS> as LaunchArg>::RuntimeArg<'a, R>;

/// Output runtime argument
pub type OutputRuntimeArg<'a, MS, R> = <OutputArg<MS> as LaunchArg>::RuntimeArg<'a, R>;

type EG<MS> = <MS as MatmulSpec>::EG;
type Args<MS> = <MS as MatmulSpec>::Args;

/// Specification for a simple standard matmul using global tensor as inputs.
#[derive(Clone)]
pub struct SingleMatmulSpec<EG: Numeric, ES: Numeric, EA: Numeric> {
    _eg: PhantomData<EG>,
    _es: PhantomData<ES>,
    _ea: PhantomData<EA>,
}

impl<EG: Numeric, ES: Numeric, EA: Numeric> MatmulSpec for SingleMatmulSpec<EG, ES, EA> {
    type EG = EG;
    type ES = ES;
    type EA = EA;
    type Args = TensorArgs;
}
