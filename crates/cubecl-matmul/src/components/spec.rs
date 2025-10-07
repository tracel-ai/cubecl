use cubecl_core::{ir::StorageType, prelude::*};
use half::{bf16, f16};

use super::global::args::{MatmulArgs, TensorArgs};

/// Matrix multiplication spec defining each element types used in the computation as well as
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
    /// Element type of lhs input tensor of the kernel.
    type Lhs: MatrixPrecision;
    /// Element type of rhs input tensor of the kernel.
    type Rhs: MatrixPrecision;
    /// Element type of acc input tensor of the kernel.
    type Acc: MatrixPrecision;
}

pub trait MatrixPrecision: Send + Sync + Copy + 'static {
    /// Element type of input tensor in global memory
    type Global: Numeric;
    /// Element type once stored in shared memory
    type Stage: Numeric;
    /// Element type once in registers for computation
    type Register: Numeric;
}

impl<EG: Numeric, ES: Numeric> MatrixPrecision for (EG, ES) {
    type Global = EG;
    type Stage = ES;
    type Register = ES;
}

impl MatmulPrecision for f16 {
    type Lhs = (f16, f16);
    type Rhs = (f16, f16);
    #[cfg(target_os = "macos")]
    type Acc = (f16, f16);
    #[cfg(not(target_os = "macos"))]
    type Acc = (f16, f32);
}

impl MatmulPrecision for flex32 {
    type Lhs = (f32, f16);
    type Rhs = (f32, f16);
    type Acc = (f32, f32);
}

impl MatmulPrecision for bf16 {
    type Lhs = (bf16, bf16);
    type Rhs = (bf16, bf16);
    #[cfg(target_os = "macos")]
    type Acc = (bf16, bf16);
    #[cfg(not(target_os = "macos"))]
    type Acc = (bf16, f32);
}

impl MatmulPrecision for f32 {
    type Lhs = (f32, f32);
    type Rhs = (f32, f32);
    type Acc = (f32, f32);
}

impl MatmulPrecision for f64 {
    type Lhs = (f64, f32);
    type Rhs = (f64, f32);
    type Acc = (f64, f32);
}

impl MatmulPrecision for u8 {
    type Lhs = (u8, u8);
    type Rhs = (u8, u8);
    type Acc = (i32, i32);
}

impl MatmulPrecision for u16 {
    type Lhs = (u16, u16);
    type Rhs = (u16, u16);
    type Acc = (i32, i32);
}

impl MatmulPrecision for u32 {
    type Lhs = (u32, u32);
    type Rhs = (u32, u32);
    type Acc = (u32, u32);
}

impl MatmulPrecision for u64 {
    type Lhs = (u64, u64);
    type Rhs = (u64, u64);
    type Acc = (u64, u64);
}

impl MatmulPrecision for i8 {
    type Lhs = (i8, i8);
    type Rhs = (i8, i8);
    type Acc = (i32, i32);
}

impl MatmulPrecision for i16 {
    type Lhs = (i16, i16);
    type Rhs = (i16, i16);
    type Acc = (i32, i32);
}

impl MatmulPrecision for i32 {
    type Lhs = (i32, i32);
    type Rhs = (i32, i32);
    type Acc = (i32, i32);
}

impl MatmulPrecision for i64 {
    type Lhs = (i64, i64);
    type Rhs = (i64, i64);
    type Acc = (i64, i64);
}

impl<LhsG: Numeric, RhsG: Numeric, AccG: Numeric, LhsS: Numeric, RhsS: Numeric, AccS: Numeric>
    MatmulPrecision for (LhsG, RhsG, AccG, LhsS, RhsS, AccS)
{
    type Lhs = (LhsG, LhsS);
    type Rhs = (RhsG, RhsS);
    type Acc = (AccG, AccS);
}

/// Input argument
pub type InputArg<MS> = <Args<MS> as MatmulArgs>::Input<LhsG<MS>, RhsG<MS>, AccG<MS>>;

/// Output argument
pub type OutputArg<MS> = <Args<MS> as MatmulArgs>::Output<AccG<MS>>;

/// Input runtime argument
pub type InputRuntimeArg<'a, MS, R> = <InputArg<MS> as LaunchArg>::RuntimeArg<'a, R>;

/// Output runtime argument
pub type OutputRuntimeArg<'a, MS, R> = <OutputArg<MS> as LaunchArg>::RuntimeArg<'a, R>;

pub type LhsG<MS> =
    <<<MS as MatmulSpec>::Precision as MatmulPrecision>::Lhs as MatrixPrecision>::Global;
pub type LhsS<MS> =
    <<<MS as MatmulSpec>::Precision as MatmulPrecision>::Lhs as MatrixPrecision>::Stage;
pub type LhsR<MS> =
    <<<MS as MatmulSpec>::Precision as MatmulPrecision>::Lhs as MatrixPrecision>::Register;
pub type RhsG<MS> =
    <<<MS as MatmulSpec>::Precision as MatmulPrecision>::Rhs as MatrixPrecision>::Global;
pub type RhsS<MS> =
    <<<MS as MatmulSpec>::Precision as MatmulPrecision>::Rhs as MatrixPrecision>::Stage;
pub type RhsR<MS> =
    <<<MS as MatmulSpec>::Precision as MatmulPrecision>::Rhs as MatrixPrecision>::Register;
pub type AccG<MS> =
    <<<MS as MatmulSpec>::Precision as MatmulPrecision>::Acc as MatrixPrecision>::Global;
pub type AccS<MS> =
    <<<MS as MatmulSpec>::Precision as MatmulPrecision>::Acc as MatrixPrecision>::Stage;
pub type AccR<MS> =
    <<<MS as MatmulSpec>::Precision as MatmulPrecision>::Acc as MatrixPrecision>::Register;

pub type Args<MS> = <MS as MatmulSpec>::Args;

pub struct MatmulElems {
    pub lhs_global: StorageType,
    pub rhs_global: StorageType,
    pub acc_global: StorageType,
    pub lhs_stage: StorageType,
    pub rhs_stage: StorageType,
    pub acc_stage: StorageType,
    pub lhs_register: StorageType,
    pub rhs_register: StorageType,
    pub acc_register: StorageType,
}

impl MatmulElems {
    pub fn new<MP: MatmulPrecision>() -> Self {
        Self {
            lhs_global: <MP::Lhs as MatrixPrecision>::Global::as_type_native_unchecked(),
            rhs_global: <MP::Rhs as MatrixPrecision>::Global::as_type_native_unchecked(),
            acc_global: <MP::Acc as MatrixPrecision>::Global::as_type_native_unchecked(),
            lhs_stage: <MP::Lhs as MatrixPrecision>::Stage::as_type_native_unchecked(),
            rhs_stage: <MP::Rhs as MatrixPrecision>::Stage::as_type_native_unchecked(),
            acc_stage: <MP::Acc as MatrixPrecision>::Stage::as_type_native_unchecked(),
            lhs_register: <MP::Lhs as MatrixPrecision>::Register::as_type_native_unchecked(),
            rhs_register: <MP::Rhs as MatrixPrecision>::Register::as_type_native_unchecked(),
            acc_register: <MP::Acc as MatrixPrecision>::Register::as_type_native_unchecked(),
        }
    }
}
