use cubecl_core::{ir::StorageType, prelude::*};
use half::{bf16, f16};

use crate::{
    components::{MatmulIdent, tile::TileMatmulFamily},
    tune_key::MatmulElemType,
};

use super::global::args::MatmulArgs;

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

impl<EG: Numeric, ES: Numeric, ER: Numeric> MatrixPrecision for (EG, ES, ER) {
    type Global = EG;
    type Stage = ES;
    type Register = ER;
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

impl<Lhs: MatrixPrecision, Rhs: MatrixPrecision, Acc: MatrixPrecision> MatmulPrecision
    for (Lhs, Rhs, Acc)
{
    type Lhs = Lhs;
    type Rhs = Rhs;
    type Acc = Acc;
}

pub type LhsG<MP> = <<MP as MatmulPrecision>::Lhs as MatrixPrecision>::Global;
pub type LhsS<MP> = <<MP as MatmulPrecision>::Lhs as MatrixPrecision>::Stage;
pub type LhsR<MP> = <<MP as MatmulPrecision>::Lhs as MatrixPrecision>::Register;

pub type RhsG<MP> = <<MP as MatmulPrecision>::Rhs as MatrixPrecision>::Global;
pub type RhsS<MP> = <<MP as MatmulPrecision>::Rhs as MatrixPrecision>::Stage;
pub type RhsR<MP> = <<MP as MatmulPrecision>::Rhs as MatrixPrecision>::Register;

pub type AccG<MP> = <<MP as MatmulPrecision>::Acc as MatrixPrecision>::Global;
pub type AccS<MP> = <<MP as MatmulPrecision>::Acc as MatrixPrecision>::Stage;
pub type AccR<MP> = <<MP as MatmulPrecision>::Acc as MatrixPrecision>::Register;

/// Input argument
pub type InputArg<MA> =
    <MA as MatmulArgs>::Input<NumericExpand<0>, NumericExpand<1>, NumericExpand<2>>;

/// Output argument
pub type OutputArg<MA> = <MA as MatmulArgs>::Output<NumericExpand<2>>;

/// Input runtime argument
pub type InputRuntimeArg<'a, MA, R> = <InputArg<MA> as LaunchArg>::RuntimeArg<'a, R>;

/// Output runtime argument
pub type OutputRuntimeArg<'a, MA, R> = <OutputArg<MA> as LaunchArg>::RuntimeArg<'a, R>;

#[derive(Clone, Debug)]
pub struct MatmulElems {
    pub lhs_global: MatmulElemType,
    pub rhs_global: MatmulElemType,
    pub acc_global: MatmulElemType,
    pub lhs_stage: MatmulElemType,
    pub rhs_stage: MatmulElemType,
    pub acc_stage: MatmulElemType,
    pub lhs_register: MatmulElemType,
    pub rhs_register: MatmulElemType,
    pub acc_register: MatmulElemType,
}

impl MatmulElems {
    pub fn new<MP: MatmulPrecision>() -> Self {
        Self {
            lhs_global: MatmulElemType::new(
                <MP::Lhs as MatrixPrecision>::Global::as_type_native_unchecked(),
                false,
            ),
            rhs_global: MatmulElemType::new(
                <MP::Rhs as MatrixPrecision>::Global::as_type_native_unchecked(),
                false,
            ),
            acc_global: MatmulElemType::new(
                <MP::Acc as MatrixPrecision>::Global::as_type_native_unchecked(),
                false,
            ),
            lhs_stage: MatmulElemType::new(
                <MP::Lhs as MatrixPrecision>::Stage::as_type_native_unchecked(),
                false,
            ),
            rhs_stage: MatmulElemType::new(
                <MP::Rhs as MatrixPrecision>::Stage::as_type_native_unchecked(),
                false,
            ),
            acc_stage: MatmulElemType::new(
                <MP::Acc as MatrixPrecision>::Stage::as_type_native_unchecked(),
                false,
            ),
            lhs_register: MatmulElemType::new(
                <MP::Lhs as MatrixPrecision>::Register::as_type_native_unchecked(),
                false,
            ),
            rhs_register: MatmulElemType::new(
                <MP::Rhs as MatrixPrecision>::Register::as_type_native_unchecked(),
                false,
            ),
            acc_register: MatmulElemType::new(
                <MP::Acc as MatrixPrecision>::Register::as_type_native_unchecked(),
                false,
            ),
        }
    }

    pub fn new_with_tile<MP: MatmulPrecision, TMM: TileMatmulFamily>() -> Self {
        fn stage<MP: MatrixPrecision, TMM: TileMatmulFamily>() -> MatmulElemType {
            MatmulElemType::new(
                if TMM::can_cast_stage_element() {
                    MP::Global::as_type_native_unchecked()
                } else {
                    MP::Register::as_type_native_unchecked()
                },
                false,
            )
        }

        Self {
            lhs_global: MatmulElemType::new(
                <MP::Lhs as MatrixPrecision>::Global::as_type_native_unchecked(),
                false,
            ),
            rhs_global: MatmulElemType::new(
                <MP::Rhs as MatrixPrecision>::Global::as_type_native_unchecked(),
                false,
            ),
            acc_global: MatmulElemType::new(
                <MP::Acc as MatrixPrecision>::Global::as_type_native_unchecked(),
                false,
            ),
            lhs_stage: stage::<MP::Lhs, TMM>(),
            rhs_stage: stage::<MP::Rhs, TMM>(),
            acc_stage: stage::<MP::Acc, TMM>(),
            lhs_register: MatmulElemType::new(
                <MP::Lhs as MatrixPrecision>::Register::as_type_native_unchecked(),
                false,
            ),
            rhs_register: MatmulElemType::new(
                <MP::Rhs as MatrixPrecision>::Register::as_type_native_unchecked(),
                false,
            ),
            acc_register: MatmulElemType::new(
                <MP::Acc as MatrixPrecision>::Register::as_type_native_unchecked(),
                false,
            ),
        }
    }

    pub fn from_globals(lhs: MatmulElemType, rhs: MatmulElemType, out: MatmulElemType) -> Self {
        let acc_type = |dtype: StorageType| {
            if dtype == half::f16::as_type_native_unchecked()
                || dtype == half::bf16::as_type_native_unchecked()
            {
                return MatmulElemType::new(f32::as_type_native_unchecked(), false);
            }

            MatmulElemType::new(dtype, false)
        };

        Self {
            lhs_global: lhs,
            rhs_global: rhs,
            acc_global: out,
            lhs_stage: MatmulElemType::new(lhs.dtype, false),
            rhs_stage: MatmulElemType::new(rhs.dtype, false),
            acc_stage: acc_type(out.dtype),
            lhs_register: MatmulElemType::new(lhs.dtype, false),
            rhs_register: MatmulElemType::new(rhs.dtype, false),
            acc_register: acc_type(out.dtype),
        }
    }

    pub fn global(&self, ident: MatmulIdent) -> MatmulElemType {
        match ident {
            MatmulIdent::Lhs => self.lhs_global,
            MatmulIdent::Rhs => self.rhs_global,
            MatmulIdent::Out => self.acc_global,
        }
    }

    pub fn stage(&self, ident: MatmulIdent) -> MatmulElemType {
        match ident {
            MatmulIdent::Lhs => self.lhs_stage,
            MatmulIdent::Rhs => self.rhs_stage,
            MatmulIdent::Out => self.acc_stage,
        }
    }

    pub fn register(&self, ident: MatmulIdent) -> MatmulElemType {
        match ident {
            MatmulIdent::Lhs => self.lhs_register,
            MatmulIdent::Rhs => self.rhs_register,
            MatmulIdent::Out => self.acc_register,
        }
    }
}
