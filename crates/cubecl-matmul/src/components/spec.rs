use cubecl_core::{ir::Elem, prelude::*};
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
    /// Element type of each input tensors of the kernel.
    type Lhs: InputPrecision;
    /// Element type for the shared memories used to read inputs.
    type Rhs: InputPrecision;
    /// Element type for the shared memories or fragments used to accumulate
    /// smaller matmul results before writing to the output tensor.
    type EA: Numeric;
    /// Element type of the output tensor of the kernel.
    type EO: Numeric;
}

pub trait InputPrecision: Send + Sync + Copy + 'static {
    /// Element type of input tensor in global memory
    type Global: Numeric;
    /// Element type once stored in shared memory
    type Stage: Numeric;
    /// Element type once in registers for computation
    type Register: Numeric;
}

impl<EG: Numeric, ES: Numeric> InputPrecision for (EG, ES) {
    type Global = EG;
    type Stage = ES;
    type Register = ES;
}

impl MatmulPrecision for f16 {
    type Lhs = (f16, f16);
    type Rhs = (f16, f16);
    #[cfg(target_os = "macos")]
    type EA = f16;
    #[cfg(not(target_os = "macos"))]
    type EA = f32;
    type EO = f16;
}

impl MatmulPrecision for flex32 {
    type Lhs = (f32, f16);
    type Rhs = (f32, f16);
    type EA = f32;
    type EO = f32;
}

impl MatmulPrecision for bf16 {
    type Lhs = (bf16, bf16);
    type Rhs = (bf16, bf16);
    #[cfg(target_os = "macos")]
    type EA = bf16;
    #[cfg(not(target_os = "macos"))]
    type EA = f32;
    type EO = bf16;
}

impl MatmulPrecision for f32 {
    type Lhs = (f32, f32);
    type Rhs = (f32, f32);
    type EA = f32;
    type EO = f32;
}

impl MatmulPrecision for f64 {
    type Lhs = (f64, f32);
    type Rhs = (f64, f32);
    type EA = f32;
    type EO = f64;
}

impl<LhsG: Numeric, RhsG: Numeric, LhsS: Numeric, RhsS: Numeric, EA: Numeric, EO: Numeric>
    MatmulPrecision for (LhsG, RhsG, LhsS, RhsS, EA, EO)
{
    type Lhs = (LhsG, LhsS);
    type Rhs = (RhsG, RhsS);
    type EA = EA;
    type EO = EO;
}

/// Input argument
pub type InputArg<MS> = <Args<MS> as MatmulArgs>::Input<LhsG<MS>, RhsG<MS>>;

/// Output argument
pub type OutputArg<MS> = <Args<MS> as MatmulArgs>::Output<EO<MS>>;

/// Input runtime argument
pub type InputRuntimeArg<'a, MS, R> = <InputArg<MS> as LaunchArg>::RuntimeArg<'a, R>;

/// Output runtime argument
pub type OutputRuntimeArg<'a, MS, R> = <OutputArg<MS> as LaunchArg>::RuntimeArg<'a, R>;

pub type LhsG<MS> =
    <<<MS as MatmulSpec>::Precision as MatmulPrecision>::Lhs as InputPrecision>::Global;
pub type LhsS<MS> =
    <<<MS as MatmulSpec>::Precision as MatmulPrecision>::Lhs as InputPrecision>::Stage;
pub type LhsR<MS> =
    <<<MS as MatmulSpec>::Precision as MatmulPrecision>::Lhs as InputPrecision>::Register;
pub type RhsG<MS> =
    <<<MS as MatmulSpec>::Precision as MatmulPrecision>::Rhs as InputPrecision>::Global;
pub type RhsS<MS> =
    <<<MS as MatmulSpec>::Precision as MatmulPrecision>::Rhs as InputPrecision>::Stage;
pub type RhsR<MS> =
    <<<MS as MatmulSpec>::Precision as MatmulPrecision>::Rhs as InputPrecision>::Register;

pub type EA<MS> = <<MS as MatmulSpec>::Precision as MatmulPrecision>::EA;
pub type EO<MS> = <<MS as MatmulSpec>::Precision as MatmulPrecision>::EO;

pub type Args<MS> = <MS as MatmulSpec>::Args;

pub struct MatmulElems {
    pub lhs_global: Elem,
    pub rhs_global: Elem,
    pub lhs_stage: Elem,
    pub rhs_stage: Elem,
    pub lhs_register: Elem,
    pub rhs_register: Elem,
    pub acc: Elem,
    pub out: Elem,
}

impl MatmulElems {
    pub fn new<MP: MatmulPrecision>() -> Self {
        Self {
            lhs_global: <MP::Lhs as InputPrecision>::Global::as_elem_native_unchecked(),
            rhs_global: <MP::Rhs as InputPrecision>::Global::as_elem_native_unchecked(),
            lhs_stage: <MP::Lhs as InputPrecision>::Stage::as_elem_native_unchecked(),
            rhs_stage: <MP::Rhs as InputPrecision>::Stage::as_elem_native_unchecked(),
            lhs_register: <MP::Lhs as InputPrecision>::Register::as_elem_native_unchecked(),
            rhs_register: <MP::Rhs as InputPrecision>::Register::as_elem_native_unchecked(),
            acc: MP::EA::as_elem_native_unchecked(),
            out: MP::EO::as_elem_native_unchecked(),
        }
    }
}
