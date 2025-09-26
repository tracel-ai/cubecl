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

pub trait QueryPrecision: Send + Sync + Copy + 'static {
    type Global: Float;
    type Tile: Float;
}

pub trait KeyValuePrecision: Send + Sync + Copy + 'static {
    type Global: Float;
    type Stage: Float;
    type Tile: Float;
}

pub trait AttentionPrecision: Send + Sync + Copy + 'static {
    type Query: QueryPrecision;
    type Key: KeyValuePrecision;
    type Value: KeyValuePrecision;
    type Softmax: Float;
    type Accumulator: Float;
    type Mask: Numeric;
    type Out: Float;
}

impl QueryPrecision for f16 {
    type Global = f16;
    type Tile = f16;
}

impl QueryPrecision for bf16 {
    type Global = bf16;
    type Tile = bf16;
}

impl QueryPrecision for flex32 {
    type Global = f32;
    type Tile = f16;
}

impl QueryPrecision for f32 {
    type Global = f32;
    type Tile = f32;
}

impl QueryPrecision for f64 {
    type Global = f64;
    type Tile = f32;
}

impl KeyValuePrecision for f16 {
    type Global = f16;
    type Stage = f16;
    type Tile = f16;
}

impl KeyValuePrecision for bf16 {
    type Global = bf16;
    type Stage = bf16;
    type Tile = bf16;
}

impl KeyValuePrecision for flex32 {
    type Global = f32;
    type Stage = f16;
    type Tile = f16;
}

impl KeyValuePrecision for f32 {
    type Global = f32;
    type Stage = f32;
    type Tile = f32;
}

impl KeyValuePrecision for f64 {
    type Global = f64;
    type Stage = f32;
    type Tile = f32;
}

impl AttentionPrecision for f16 {
    type Query = f16;
    type Key = f16;
    type Value = f16;
    #[cfg(target_os = "macos")]
    type Softmax = f16;
    #[cfg(target_os = "macos")]
    type Accumulator = f16;
    #[cfg(not(target_os = "macos"))]
    type Softmax = f32;
    #[cfg(not(target_os = "macos"))]
    type Softmax = f32;
    type Mask = u8;
    type Out = f16;
}

impl AttentionPrecision for flex32 {
    type Query = flex32;
    type Key = flex32;
    type Value = flex32;
    #[cfg(target_os = "macos")]
    type Softmax = f16;
    #[cfg(target_os = "macos")]
    type Accumulator = f16;
    #[cfg(not(target_os = "macos"))]
    type Softmax = f32;
    #[cfg(not(target_os = "macos"))]
    type Softmax = f32;
    type Mask = u8;
    type Out = f32;
}

impl AttentionPrecision for bf16 {
    type Query = bf16;
    type Key = bf16;
    type Value = bf16;
    #[cfg(target_os = "macos")]
    type Softmax = bf16;
    #[cfg(target_os = "macos")]
    type Accumulator = bf16;
    #[cfg(not(target_os = "macos"))]
    type Softmax = f32;
    #[cfg(not(target_os = "macos"))]
    type Softmax = f32;
    type Mask = u8;
    type Out = bf16;
}

impl AttentionPrecision for f32 {
    type Query = f32;
    type Key = f32;
    type Value = f32;
    type Softmax = f32;
    type Accumulator = f32;
    type Mask = u8;
    type Out = f32;
}

impl AttentionPrecision for f64 {
    type Query = f64;
    type Key = f64;
    type Value = f64;
    type Softmax = f32;
    type Accumulator = f32;
    type Mask = u8;
    type Out = f64;
}

/// Input argument
pub type InputArg<AS> = <Args<AS> as AttentionArgs>::Input<QG<AS>, KG<AS>, VG<AS>>;

/// Output argument
pub type OutputArg<AS> = <Args<AS> as AttentionArgs>::Output<OG<AS>>;

/// Input runtime argument
pub type InputRuntimeArg<'a, MS, R> = <InputArg<MS> as LaunchArg>::RuntimeArg<'a, R>;

/// Output runtime argument
pub type OutputRuntimeArg<'a, MS, R> = <OutputArg<MS> as LaunchArg>::RuntimeArg<'a, R>;

pub type QG<AS> =
    <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Query as QueryPrecision>::Global;
pub type QT<AS> =
    <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Query as QueryPrecision>::Tile;
pub type KG<AS> =
    <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Key as KeyValuePrecision>::Global;
pub type KS<AS> =
    <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Key as KeyValuePrecision>::Stage;
pub type KT<AS> =
    <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Key as KeyValuePrecision>::Tile;
pub type VG<AS> =
    <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Value as KeyValuePrecision>::Global;
pub type VS<AS> =
    <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Value as KeyValuePrecision>::Stage;
pub type VT<AS> =
    <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Value as KeyValuePrecision>::Tile;
pub type SM<AS> = <<AS as AttentionSpec>::Precision as AttentionPrecision>::Softmax;
pub type ACC<AS> = <<AS as AttentionSpec>::Precision as AttentionPrecision>::Accumulator;
pub type MSK<AS> = <<AS as AttentionSpec>::Precision as AttentionPrecision>::Mask;
pub type OG<AS> = <<AS as AttentionSpec>::Precision as AttentionPrecision>::Out;

pub type Args<MS> = <MS as AttentionSpec>::Args;
