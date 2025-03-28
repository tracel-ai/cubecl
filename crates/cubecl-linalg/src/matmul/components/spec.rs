use core::marker::PhantomData;

use cubecl_core::prelude::*;
use cubecl_std::SymQ8;
use half::{bf16, f16};

use super::global::args::{MatmulArgs, TensorArgs};

/// Matrix multiplication spec definiting each element types used in the computation as well as
/// how the arguments are passed to the kernel.
pub trait MatmulSpec: Send + Sync + Clone + 'static {
    type Precision: MatmulPrecision;
    /// How the input and output tensors are passed as arguments.
    type Args: MatmulArgs;
}

impl<MP: MatmulPrecision, Args: MatmulArgs> MatmulSpec for (MP, Args) {
    type Precision = MP;
    type Args = Args;
}

// A simple default for TensorArgs
impl<MP: MatmulPrecision> MatmulSpec for MP {
    type Precision = MP;
    type Args = TensorArgs;
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

impl MatmulPrecision for f16 {
    const QUANTIZED: bool = false;
    type EG = f16;
    type ES = f16;
    type EA = f32;
}

impl MatmulPrecision for flex32 {
    const QUANTIZED: bool = false;
    type EG = flex32;
    type ES = f16;
    type EA = f32;
}

impl MatmulPrecision for bf16 {
    const QUANTIZED: bool = false;
    type EG = bf16;
    type ES = bf16;
    type EA = f32;
}

impl MatmulPrecision for f32 {
    const QUANTIZED: bool = false;
    type EG = f32;
    type ES = f32;
    type EA = f32;
}

#[derive(Clone)]
pub struct ReplaceES<MP: MatmulPrecision, ES: Numeric> {
    _phantom: PhantomData<(ES, MP)>,
}

impl<MP: MatmulPrecision, ES: Numeric> MatmulPrecision for ReplaceES<MP, ES> {
    const QUANTIZED: bool = MP::QUANTIZED;
    type EG = MP::EG;
    type ES = ES;
    type EA = MP::EA;
}

impl<EG: Numeric, ES: Numeric, EA: Numeric> MatmulPrecision for (EG, ES, EA) {
    const QUANTIZED: bool = false;
    type EG = EG;
    type ES = ES;
    type EA = EA;
}

#[derive(Clone)]
pub struct Quantized;

impl<EG: Numeric, ES: Numeric, EA: Numeric> MatmulPrecision for (EG, ES, EA, Quantized) {
    const QUANTIZED: bool = true;
    type EG = EG;
    type ES = ES;
    type EA = EA;
}

impl MatmulPrecision for SymQ8 {
    const QUANTIZED: bool = true;
    type EG = i8;
    type ES = i8;
    type EA = i32;
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
