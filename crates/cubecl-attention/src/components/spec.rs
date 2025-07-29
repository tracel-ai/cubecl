use core::marker::PhantomData;

use cubecl_core::prelude::*;
use cubecl_matmul::components::MatmulPrecision;
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
    type EI: Float;
    /// Element of mask
    type EM: Numeric;
    /// Element type for the shared memories used to read inputs.
    type ES: Float;
    /// Element type for the shared memories or fragments used to accumulate
    /// smaller matmul results before writing to the output tensor.
    type EA: Float;
    /// Element type of the output tensor of the kernel.
    type EO: Float;

    type MatmulPrecision: MatmulPrecision<EI = Self::EI, ES = Self::ES, EA = Self::EA, EO = Self::EO>;
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

    type MatmulPrecision = Self;
}

impl AttentionPrecision for flex32 {
    type EI = f32;
    type EM = u8;
    type ES = f16;
    type EA = f32;
    type EO = f32;
    type MatmulPrecision = Self;
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
    type MatmulPrecision = Self;
}

impl AttentionPrecision for f32 {
    type EI = f32;
    type EM = u8;
    type ES = f32;
    type EA = f32;
    type EO = f32;
    type MatmulPrecision = Self;
}

impl AttentionPrecision for f64 {
    type EI = f64;
    type EM = u8;
    type ES = f32;
    type EA = f32;
    type EO = f64;
    type MatmulPrecision = Self;
}

#[derive(Clone, Copy)]
pub struct ReplaceES<MP: AttentionPrecision, ES: Float> {
    _phantom: PhantomData<(ES, MP)>,
}

impl<AP: AttentionPrecision, ES: Float> AttentionPrecision for ReplaceES<AP, ES> {
    type EI = AP::EI;
    type EM = AP::EM;
    type ES = ES;
    type EA = AP::EA;
    type EO = AP::EO;
    type MatmulPrecision = (Self::EI, ES, Self::EA, Self::EO);
}

impl<EI: Float, EM: Numeric, ES: Float, EA: Float, EO: Float> AttentionPrecision
    for (EI, EM, ES, EA, EO)
{
    type EI = EI;
    type EM = EM;
    type ES = ES;
    type EA = EA;
    type EO = EO;
    type MatmulPrecision = (EI, ES, EA, EO);
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
