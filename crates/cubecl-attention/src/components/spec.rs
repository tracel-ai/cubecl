use cubecl_core::prelude::*;
use half::{bf16, f16};

use crate::components::{
    args::{AttentionArgs, TensorArgs},
    spec::attention_types::*,
};

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
}

pub trait AttentionPrecision: Send + Sync + Copy + 'static {
    type Query: QueryPrecision;
    type Key: KeyValuePrecision;
    type Value: KeyValuePrecision;
    type KVTile: Float;
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

impl<G: Float, T: Float> QueryPrecision for (G, T) {
    type Global = G;
    type Tile = T;
}

impl KeyValuePrecision for f16 {
    type Global = f16;
    type Stage = f16;
}

impl KeyValuePrecision for bf16 {
    type Global = bf16;
    type Stage = bf16;
}

impl KeyValuePrecision for flex32 {
    type Global = f32;
    type Stage = f16;
}

impl KeyValuePrecision for f32 {
    type Global = f32;
    type Stage = f32;
}

impl KeyValuePrecision for f64 {
    type Global = f64;
    type Stage = f32;
}

impl<G: Float, S: Float> KeyValuePrecision for (G, S) {
    type Global = G;
    type Stage = S;
}

impl AttentionPrecision for f16 {
    type Query = f16;
    type Key = f16;
    type Value = f16;
    type KVTile = f16;
    #[cfg(target_os = "macos")]
    type Softmax = f16;
    #[cfg(target_os = "macos")]
    type Accumulator = f16;
    #[cfg(not(target_os = "macos"))]
    type Softmax = f32;
    #[cfg(not(target_os = "macos"))]
    type Accumulator = f32;
    type Mask = u8;
    type Out = f16;
}

impl AttentionPrecision for flex32 {
    type Query = flex32;
    type Key = flex32;
    type Value = flex32;
    type KVTile = f16;
    #[cfg(target_os = "macos")]
    type Softmax = f16;
    #[cfg(target_os = "macos")]
    type Accumulator = f16;
    #[cfg(not(target_os = "macos"))]
    type Softmax = f32;
    #[cfg(not(target_os = "macos"))]
    type Accumulator = f32;
    type Mask = u8;
    type Out = f32;
}

impl AttentionPrecision for bf16 {
    type Query = bf16;
    type Key = bf16;
    type Value = bf16;
    type KVTile = bf16;
    #[cfg(target_os = "macos")]
    type Softmax = bf16;
    #[cfg(target_os = "macos")]
    type Accumulator = bf16;
    #[cfg(not(target_os = "macos"))]
    type Softmax = f32;
    #[cfg(not(target_os = "macos"))]
    type Accumulator = f32;
    type Mask = u8;
    type Out = bf16;
}

impl AttentionPrecision for f32 {
    type Query = f32;
    type Key = f32;
    type Value = f32;
    type KVTile = f32;
    type Softmax = f32;
    type Accumulator = f32;
    type Mask = u8;
    type Out = f32;
}

impl AttentionPrecision for f64 {
    type Query = f64;
    type Key = f64;
    type Value = f64;
    type KVTile = f32;
    type Softmax = f32;
    type Accumulator = f32;
    type Mask = u8;
    type Out = f64;
}

impl<
    QG: Float,
    QT: Float,
    KG: Float,
    KS: Float,
    VG: Float,
    VS: Float,
    KVT: Float,
    SM: Float,
    ACC: Float,
    MSK: Numeric,
    OG: Float,
> AttentionPrecision for (QG, QT, KG, KS, VG, VS, KVT, SM, ACC, MSK, OG)
{
    type Query = (QG, QT);
    type Key = (KG, KS);
    type Value = (VG, VS);
    type KVTile = KVT;
    type Softmax = SM;
    type Accumulator = ACC;
    type Mask = MSK;
    type Out = OG;
}

/// Input argument
pub type InputArg<AS> = <Args<AS> as AttentionArgs>::Input<QG<AS>, KG<AS>, VG<AS>>;

/// Output argument
pub type OutputArg<AS> = <Args<AS> as AttentionArgs>::Output<OG<AS>>;

/// Input runtime argument
pub type InputRuntimeArg<'a, MS, R> = <InputArg<MS> as LaunchArg>::RuntimeArg<'a, R>;

/// Output runtime argument
pub type OutputRuntimeArg<'a, MS, R> = <OutputArg<MS> as LaunchArg>::RuntimeArg<'a, R>;

pub mod attention_types {
    use crate::components::{AttentionPrecision, AttentionSpec, KeyValuePrecision, QueryPrecision};

    pub type QG<AS> =
        <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Query as QueryPrecision>::Global;
    pub type QT<AS> =
        <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Query as QueryPrecision>::Tile;
    pub type KG<AS> =
    <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Key as KeyValuePrecision>::Global;
    pub type KS<AS> =
        <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Key as KeyValuePrecision>::Stage;
    pub type VG<AS> =
    <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Value as KeyValuePrecision>::Global;
    pub type VS<AS> =
    <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Value as KeyValuePrecision>::Stage;

    pub type KVT<AS> = <<AS as AttentionSpec>::Precision as AttentionPrecision>::KVTile;
    pub type SM<AS> = <<AS as AttentionSpec>::Precision as AttentionPrecision>::Softmax;
    pub type ACC<AS> = <<AS as AttentionSpec>::Precision as AttentionPrecision>::Accumulator;
    pub type MSK<AS> = <<AS as AttentionSpec>::Precision as AttentionPrecision>::Mask;
    pub type OG<AS> = <<AS as AttentionSpec>::Precision as AttentionPrecision>::Out;

    // TODO
    pub type OS<AS> = <<AS as AttentionSpec>::Precision as AttentionPrecision>::Out;
}

pub type Args<MS> = <MS as AttentionSpec>::Args;
