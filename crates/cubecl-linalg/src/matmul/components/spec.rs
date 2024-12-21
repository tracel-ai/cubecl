use std::marker::PhantomData;

use cubecl_core::prelude::{FloatExpand, LaunchArg, Numeric, TypeMap};

use super::global::args::{MatmulArgs, TensorArgs};

/// Matrix multiplication spec definiting each element types used in the computation as well as
/// how the arguments are passed to the kernel.
pub trait MatmulSpec: Send + Sync + Clone + 'static {
    /// The plane size used by this kernel.
    const PLANE_DIM: u32;

    /// Element type of each input and output tensor of the kernel.
    type EG: Numeric;
    /// Element type of the intermediate representation of the inputs.
    type ES: Numeric;
    /// Element type of the intermediate representation of the output accumulator.
    type EA: Numeric;
    /// How the input and output tensors are passed as arguments.
    type Args: MatmulArgs;
}

/// Input argument
pub type InputArg<MS> = <Args<MS> as MatmulArgs>::Input<EG<MS>>;

/// Output argument
pub type OutputArg<MS> = <Args<MS> as MatmulArgs>::Output<EG<MS>>;

/// Input runtime argument
pub type InputRuntimeArg<'a, MS, R> = <InputArg<MS> as LaunchArg>::RuntimeArg<'a, R>;

/// Output runtime argument
pub type OutputRuntimeArg<'a, MS, R> = <OutputArg<MS> as LaunchArg>::RuntimeArg<'a, R>;

type EG<MS> = <MS as MatmulSpec>::EG;
type Args<MS> = <MS as MatmulSpec>::Args;

/// Specification for a simple standard matmul using global tensor as inputs.
#[derive(Clone)]
pub struct SingleMatmulSpec<const PLANE_DIM: u32, EG, ES, EA, Args = TensorArgs> {
    _eg: PhantomData<EG>,
    _es: PhantomData<ES>,
    _ea: PhantomData<EA>,
    _args: PhantomData<Args>,
}

impl<Args: MatmulArgs, EG: Numeric, ES: Numeric, EA: Numeric, const PLANE_DIM: u32> MatmulSpec
    for SingleMatmulSpec<PLANE_DIM, EG, ES, EA, Args>
{
    const PLANE_DIM: u32 = PLANE_DIM;

    type EG = EG;
    type ES = ES;
    type EA = EA;
    type Args = Args;
}

impl<const POS: u8, const PLANE_DIM: u32, EG: Numeric, ES: Numeric, EA: Numeric> TypeMap<POS>
    for SingleMatmulSpec<PLANE_DIM, EG, ES, EA>
{
    type ExpandGeneric =
        SingleMatmulSpec<PLANE_DIM, FloatExpand<0>, FloatExpand<1>, FloatExpand<2>>;

    fn register(context: &mut cubecl_core::prelude::CubeContext) {
        let eg = EG::as_elem(&context);
        let es = EG::as_elem(&context);
        let ea = EG::as_elem(&context);

        context.register_type::<FloatExpand<0>>(eg);
        context.register_type::<FloatExpand<1>>(es);
        context.register_type::<FloatExpand<2>>(ea);
    }
}
