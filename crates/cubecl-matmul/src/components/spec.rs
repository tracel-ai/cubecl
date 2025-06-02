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
pub trait MatmulPrecision: Send + Sync + Copy + 'static {
    const QUANTIZED: bool;

    /// Element type of each input tensors of the kernel.
    type EI: Numeric;
    /// Element type for the shared memories used to read inputs.
    type ES: Numeric;
    /// Element type for the shared memories or fragments used to accumulate
    /// smaller matmul results before writing to the output tensor.
    type EA: Numeric;
    /// Element type of the output tensor of the kernel.
    type EO: Numeric;
}

impl MatmulPrecision for f16 {
    const QUANTIZED: bool = false;
    type EI = f16;
    type ES = f16;
    #[cfg(target_os = "macos")]
    type EA = f16;
    #[cfg(not(target_os = "macos"))]
    type EA = f32;
    type EO = f16;
}

impl MatmulPrecision for flex32 {
    const QUANTIZED: bool = false;
    type EI = f32;
    type ES = f16;
    type EA = f32;
    type EO = f32;
}

impl MatmulPrecision for bf16 {
    const QUANTIZED: bool = false;
    type EI = bf16;
    type ES = bf16;
    #[cfg(target_os = "macos")]
    type EA = bf16;
    #[cfg(not(target_os = "macos"))]
    type EA = f32;
    type EO = bf16;
}

impl MatmulPrecision for f32 {
    const QUANTIZED: bool = false;
    type EI = f32;
    type ES = f32;
    type EA = f32;
    type EO = f32;
}

impl MatmulPrecision for f64 {
    const QUANTIZED: bool = false;
    type EI = f64;
    type ES = f32;
    type EA = f32;
    type EO = f64;
}

#[derive(Clone, Copy)]
pub struct ReplaceES<MP: MatmulPrecision, ES: Numeric> {
    _phantom: PhantomData<(ES, MP)>,
}

impl<MP: MatmulPrecision, ES: Numeric> MatmulPrecision for ReplaceES<MP, ES> {
    const QUANTIZED: bool = MP::QUANTIZED;
    type EI = MP::EI;
    type ES = ES;
    type EA = MP::EA;
    type EO = MP::EO;
}

impl<EI: Numeric, ES: Numeric, EA: Numeric, EO: Numeric> MatmulPrecision for (EI, ES, EA, EO) {
    const QUANTIZED: bool = false;
    type EI = EI;
    type ES = ES;
    type EA = EA;
    type EO = EO;
}

#[derive(Clone, Copy)]
pub struct Quantized;

impl<EI: Numeric, ES: Numeric, EA: Numeric, EO: Numeric> MatmulPrecision
    for (EI, ES, EA, EO, Quantized)
{
    const QUANTIZED: bool = true;
    type EI = EI;
    type ES = ES;
    type EA = EA;
    type EO = EO;
}

impl MatmulPrecision for SymQ8 {
    const QUANTIZED: bool = true;
    type EI = i8;
    type ES = f16;
    type EA = f16;
    type EO = f16;
}

/// Input argument
pub type InputArg<MS> = <Args<MS> as MatmulArgs>::Input<EI<MS>>;

/// Output argument
pub type OutputArg<MS> = <Args<MS> as MatmulArgs>::Output<EO<MS>>;

/// Input runtime argument
pub type InputRuntimeArg<'a, MS, R> = <InputArg<MS> as LaunchArg>::RuntimeArg<'a, R>;

/// Output runtime argument
pub type OutputRuntimeArg<'a, MS, R> = <OutputArg<MS> as LaunchArg>::RuntimeArg<'a, R>;

pub type EI<MS> = <<MS as MatmulSpec>::Precision as MatmulPrecision>::EI;
pub type ES<MS> = <<MS as MatmulSpec>::Precision as MatmulPrecision>::ES;
pub type EA<MS> = <<MS as MatmulSpec>::Precision as MatmulPrecision>::EA;
pub type EO<MS> = <<MS as MatmulSpec>::Precision as MatmulPrecision>::EO;

pub type Args<MS> = <MS as MatmulSpec>::Args;
