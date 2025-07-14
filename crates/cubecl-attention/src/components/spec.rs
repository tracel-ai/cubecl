use core::marker::PhantomData;

use cubecl_core::prelude::*;
use half::{bf16, f16};

use crate::components::args::{AttentionArgs, TensorArgs};

/// Attention spec definiting each element types used in the computation as well as
/// how the arguments are passed to the kernel.
pub trait AttentionSpec: Send + Sync + Clone + 'static {
    type Precision: AttentionPrecision;
    /// How the input and output tensors are passed as arguments.
    type Args: AttentionArgs;
}

impl<AP: AttentionPrecision, Args: AttentionArgs> AttentionSpec for (AP, Args) {
    type Precision = AP;
    type Args = Args;
}

// A simple default for TensorArgs
impl<AP: AttentionPrecision> AttentionSpec for AP {
    type Precision = AP;
    type Args = TensorArgs;
}

/// Matrix multiplication precisions.
pub trait AttentionPrecision: Send + Sync + Copy + 'static {
    /// Element type of each input tensors of the kernel.
    type EI: Numeric;
    /// Element of mask
    type EM: Numeric;
    /// Element type for the shared memories used to read inputs.
    type ES: Numeric;
    /// Element type for the shared memories or fragments used to accumulate
    /// smaller matmul results before writing to the output tensor.
    type EA: Numeric;
    /// Element type of the output tensor of the kernel.
    type EO: Numeric;
}

impl AttentionPrecision for f16 {
    type EI = f16;
    type EM = u8;
    type ES = f16;
    #[cfg(target_os = "macos")]
    type EA = f16;
    #[cfg(not(target_os = "macos"))]
    type EA = f32;
    type EO = f16;
}

impl AttentionPrecision for flex32 {
    type EI = f32;
    type EM = u8;
    type ES = f16;
    type EA = f32;
    type EO = f32;
}

impl AttentionPrecision for bf16 {
    type EI = bf16;
    type EM = u8;
    type ES = bf16;
    #[cfg(target_os = "macos")]
    type EA = bf16;
    #[cfg(not(target_os = "macos"))]
    type EA = f32;
    type EO = bf16;
}

impl AttentionPrecision for f32 {
    type EI = f32;
    type EM = u8;
    type ES = f32;
    type EA = f32;
    type EO = f32;
}

impl AttentionPrecision for f64 {
    type EI = f64;
    type EM = u8;
    type ES = f32;
    type EA = f32;
    type EO = f64;
}

#[derive(Clone, Copy)]
pub struct ReplaceES<MP: AttentionPrecision, ES: Numeric> {
    _phantom: PhantomData<(ES, MP)>,
}

impl<MP: AttentionPrecision, ES: Numeric> AttentionPrecision for ReplaceES<MP, ES> {
    type EI = MP::EI;
    type EM = MP::EM;
    type ES = ES;
    type EA = MP::EA;
    type EO = MP::EO;
}

impl<EI: Numeric, EM: Numeric, ES: Numeric, EA: Numeric, EO: Numeric> AttentionPrecision
    for (EI, EM, ES, EA, EO)
{
    type EI = EI;
    type EM = EM;
    type ES = ES;
    type EA = EA;
    type EO = EO;
}

/// Input argument
pub type InputArg<MS> = <Args<MS> as AttentionArgs>::Input<EI<MS>>;

/// Output argument
pub type OutputArg<MS> = <Args<MS> as AttentionArgs>::Output<EO<MS>>;

/// Input runtime argument
pub type InputRuntimeArg<'a, MS, R> = <InputArg<MS> as LaunchArg>::RuntimeArg<'a, R>;

/// Output runtime argument
pub type OutputRuntimeArg<'a, MS, R> = <OutputArg<MS> as LaunchArg>::RuntimeArg<'a, R>;

pub type EI<MS> = <<MS as AttentionSpec>::Precision as AttentionPrecision>::EI;
pub type EM<MS> = <<MS as AttentionSpec>::Precision as AttentionPrecision>::EM;
pub type ES<MS> = <<MS as AttentionSpec>::Precision as AttentionPrecision>::ES;
pub type EA<MS> = <<MS as AttentionSpec>::Precision as AttentionPrecision>::EA;
pub type EO<MS> = <<MS as AttentionSpec>::Precision as AttentionPrecision>::EO;

pub type Args<MS> = <MS as AttentionSpec>::Args;
