use cubecl_core::prelude::*;
use half::{bf16, f16};

use crate::components::{
    AccumulatorPrecision, AttentionProblem,
    args::{AttentionArgs, TensorArgs},
    spec::attention_types::*,
};

/// Attention spec defining each element types used in the computation as well as
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

pub trait StagedMatrixPrecision: Send + Sync + Copy + 'static {
    type Global: Float;
    type Stage: Float;
}

pub trait AttentionPrecision: Send + Sync + Copy + 'static {
    type Query: QueryPrecision;
    type Key: StagedMatrixPrecision;
    type Value: StagedMatrixPrecision;
    type KVTile: Float;
    type Softmax: Float;
    type Accumulator: Float;
    type Mask: Numeric;
    type Out: StagedMatrixPrecision;
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

impl StagedMatrixPrecision for f16 {
    type Global = f16;
    type Stage = f16;
}

impl StagedMatrixPrecision for bf16 {
    type Global = bf16;
    type Stage = bf16;
}

impl StagedMatrixPrecision for flex32 {
    type Global = f32;
    type Stage = f16;
}

impl StagedMatrixPrecision for f32 {
    type Global = f32;
    type Stage = f32;
}

impl StagedMatrixPrecision for f64 {
    type Global = f64;
    type Stage = f32;
}

impl<G: Float, S: Float> StagedMatrixPrecision for (G, S) {
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
    OS: Float,
> AttentionPrecision for (QG, QT, KG, KS, VG, VS, KVT, SM, ACC, MSK, OG, OS)
{
    type Query = (QG, QT);
    type Key = (KG, KS);
    type Value = (VG, VS);
    type KVTile = KVT;
    type Softmax = SM;
    type Accumulator = ACC;
    type Mask = MSK;
    type Out = (OG, OS);
}

// TODO make sure the numbers are the right ones

/// Input argument
pub type InputArg<AA> = <AA as AttentionArgs>::Input<
    NumericExpand<0>,
    NumericExpand<2>,
    NumericExpand<4>,
    NumericExpand<9>,
>;

/// Output argument
pub type OutputArg<AA> = <AA as AttentionArgs>::Output<NumericExpand<10>>;

/// Input runtime argument
pub type InputRuntimeArg<'a, AA, R> = <InputArg<AA> as LaunchArg>::RuntimeArg<'a, R>;

/// Output runtime argument
pub type OutputRuntimeArg<'a, AA, R> = <OutputArg<AA> as LaunchArg>::RuntimeArg<'a, R>;

pub mod attention_types {
    use crate::components::{
        AttentionPrecision, AttentionSpec, QueryPrecision, StagedMatrixPrecision,
    };

    pub type QG<AS> =
        <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Query as QueryPrecision>::Global;
    pub type QT<AS> =
        <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Query as QueryPrecision>::Tile;
    pub type KG<AS> =
    <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Key as StagedMatrixPrecision>::Global;
    pub type KS<AS> =
        <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Key as StagedMatrixPrecision>::Stage;
    pub type VG<AS> =
    <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Value as StagedMatrixPrecision>::Global;
    pub type VS<AS> =
    <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Value as StagedMatrixPrecision>::Stage;

    pub type KVT<AS> = <<AS as AttentionSpec>::Precision as AttentionPrecision>::KVTile;
    pub type SM<AS> = <<AS as AttentionSpec>::Precision as AttentionPrecision>::Softmax;
    pub type ACC<AS> = <<AS as AttentionSpec>::Precision as AttentionPrecision>::Accumulator;
    pub type MSK<AS> = <<AS as AttentionSpec>::Precision as AttentionPrecision>::Mask;

    pub type OG<AS> = <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Out as StagedMatrixPrecision>::Global;
    pub type OS<AS> = <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Out as StagedMatrixPrecision>::Stage;
}

pub type Args<MS> = <MS as AttentionSpec>::Args;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct AttentionElems {
    pub query_global: StorageType,
    pub query_tile: StorageType,
    pub key_global: StorageType,
    pub key_stage: StorageType,
    pub value_global: StorageType,
    pub value_stage: StorageType,
    pub key_value_tile: StorageType,
    pub softmax: StorageType,
    pub accumulator: StorageType,
    pub mask: StorageType,
    pub out_global: StorageType,
    pub out_stage: StorageType,
}

impl AttentionElems {
    pub fn new<AP: AttentionPrecision>() -> Self {
        Self {
            query_global: QG::<AP>::as_type_native_unchecked(),
            query_tile: QT::<AP>::as_type_native_unchecked(),
            key_global: KG::<AP>::as_type_native_unchecked(),
            key_stage: KS::<AP>::as_type_native_unchecked(),
            value_global: VG::<AP>::as_type_native_unchecked(),
            value_stage: VS::<AP>::as_type_native_unchecked(),
            key_value_tile: KVT::<AP>::as_type_native_unchecked(),
            softmax: SM::<AP>::as_type_native_unchecked(),
            accumulator: ACC::<AP>::as_type_native_unchecked(),
            mask: MSK::<AP>::as_type_native_unchecked(),
            out_global: OG::<AP>::as_type_native_unchecked(),
            out_stage: OS::<AP>::as_type_native_unchecked(),
        }
    }

    pub fn from_problem(problem: &AttentionProblem) -> AttentionElems {
        let global = problem.global_dtypes.clone();
        let accumulator = match problem.accumulator_precision {
            AccumulatorPrecision::Strict(storage_type) => storage_type,
            AccumulatorPrecision::Loose => AccumulatorPrecision::default_accumulator_type(),
        };

        Self {
            query_global: global.query,
            query_tile: global.query,
            key_global: global.key,
            key_stage: global.key,
            value_global: global.value,
            value_stage: global.value,
            key_value_tile: global.value,
            softmax: accumulator,
            accumulator,
            mask: global.mask,
            out_global: global.out,
            out_stage: global.out,
        }
    }
}

impl From<&AttentionElems> for [StorageType; 12] {
    fn from(elems: &AttentionElems) -> Self {
        [
            elems.query_global,
            elems.query_tile,
            elems.key_global,
            elems.key_stage,
            elems.value_global,
            elems.value_stage,
            elems.key_value_tile,
            elems.softmax,
            elems.accumulator,
            elems.mask,
            elems.out_global,
            elems.out_stage,
        ]
    }
}
