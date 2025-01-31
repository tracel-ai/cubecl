use std::marker::PhantomData;

use cubecl_core::prelude::{LaunchArg, Numeric};

use super::global::args::{MatmulArgs, TensorArgs};

/// Matrix multiplication spec definiting each element types used in the computation as well as
/// how the arguments are passed to the kernel.
pub trait MatmulSpec: Send + Sync + Clone + 'static {
    /// Element type of each input tensor of the kernel.
    type In: Numeric;
    /// Element type of the intermediate representation of the inputs.
    type State: Numeric;
    /// Element type of the intermediate representation of the output accumulator.
    type Acc: Numeric;
    /// Element type of the output tensor of the kernel.
    type Out: Numeric;
    /// How the input and output tensors are passed as arguments.
    type Args: MatmulArgs;
}

/// Matrix multiplication precisions.
pub trait MatmulPrecision: Send + Sync + Clone + 'static {
    /// Element type of each input tensor of the kernel.
    type In: Numeric;
    /// Element type of the intermediate representation of the inputs.
    type State: Numeric;
    /// Element type of the intermediate representation of the output accumulator.
    type Acc: Numeric;
    /// Element type of the output tensor of the kernel.
    type Out: Numeric;
}

impl<In: Numeric, State: Numeric, Acc: Numeric, Out: Numeric> MatmulPrecision
    for (In, State, Acc, Out)
{
    type In = In;
    type State = State;
    type Acc = Acc;
    type Out = Out;
}

/// Input argument
pub type InputArg<MS> = <Args<MS> as MatmulArgs>::Input<In<MS>>;

/// Output argument
pub type OutputArg<MS> = <Args<MS> as MatmulArgs>::Output<Out<MS>>;

/// Input runtime argument
pub type InputRuntimeArg<'a, MS, R> = <InputArg<MS> as LaunchArg>::RuntimeArg<'a, R>;

/// Output runtime argument
pub type OutputRuntimeArg<'a, MS, R> = <OutputArg<MS> as LaunchArg>::RuntimeArg<'a, R>;

type In<MS> = <MS as MatmulSpec>::In;
type Out<MS> = <MS as MatmulSpec>::Out;
type Args<MS> = <MS as MatmulSpec>::Args;

/// Specification for a simple standard matmul using global tensor as inputs.
#[derive(Clone)]
pub struct SingleMatmulSpec<In, State, Acc, Out = In, Args = TensorArgs> {
    _in: PhantomData<In>,
    _state: PhantomData<State>,
    _acc: PhantomData<Acc>,
    _out: PhantomData<Out>,
    _args: PhantomData<Args>,
}

impl<Args: MatmulArgs, In: Numeric, State: Numeric, Acc: Numeric, Out: Numeric> MatmulSpec
    for SingleMatmulSpec<In, State, Acc, Out, Args>
{
    type In = In;
    type State = State;
    type Acc = Acc;
    type Out = Out;
    type Args = Args;
}
