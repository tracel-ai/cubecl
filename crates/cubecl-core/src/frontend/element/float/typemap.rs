//! This module contains a configurable [element type](ElemExpand) for floats to be used during
//! kernel expansion to speed up Rust compilation.
//!
//! Expand functions don't need to be generated for different element types even if they are generic
//! over one, since the only use of numeric element types is to map to the [elem IR enum](Elem).
//!
//! This can be done dynamically using the scope instead, reducing the binary size and the
//! compilation time of kernels significantly.
//!
//! You can still have multiple element types in a single kernel, since [ElemExpand] uses const
//! generics to differentiate between float kinds.

use core::f32;
use std::{
    cmp::Ordering,
    ops::{
        BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign, Mul,
        MulAssign, Not, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign,
    },
};

use bytemuck::{Pod, Zeroable};
use cubecl_ir::ExpandElement;
use derive_more::derive::{
    Add, AddAssign, Display, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
use num_traits::{Num, NumCast, One, ToPrimitive, Zero};
use serde::Serialize;

use crate::{
    ir::{ElemType, FloatKind, Scope, Variable},
    prelude::Numeric,
};

use super::*;

#[allow(non_camel_case_types)]
#[repr(transparent)]
#[derive(
    Clone,
    Copy,
    Default,
    Serialize,
    Zeroable,
    Pod,
    PartialEq,
    PartialOrd,
    Neg,
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    RemAssign,
    Debug,
    Display,
)]

/// A fake element type that can be configured to map to any other element type.
pub struct ElemExpand<const POS: u8>(f32);

/// A fake float element type that can be configured to map to any other element type.
pub type FloatExpand<const POS: u8> = ElemExpand<POS>;

/// A fake numeric element type that can be configured to map to any other element type.
pub type NumericExpand<const POS: u8> = ElemExpand<POS>;

impl<const POS: u8> ElemExpand<POS> {
    pub const MIN_POSITIVE: Self = Self(half::f16::MIN_POSITIVE.to_f32_const());

    pub const fn from_f32(val: f32) -> Self {
        ElemExpand(val)
    }

    pub const fn from_f64(val: f64) -> Self {
        ElemExpand(val as f32)
    }

    pub const fn to_f32(self) -> f32 {
        self.0
    }

    pub const fn to_f64(self) -> f64 {
        self.0 as f64
    }

    pub fn total_cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }

    pub fn is_nan(&self) -> bool {
        self.0.is_nan()
    }
}

impl<const POS: u8> Mul for ElemExpand<POS> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        ElemExpand(self.0 * rhs.0)
    }
}

impl<const POS: u8> Div for ElemExpand<POS> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        ElemExpand(self.0 / rhs.0)
    }
}

impl<const POS: u8> Rem for ElemExpand<POS> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        ElemExpand(self.0 % rhs.0)
    }
}

impl<const POS: u8> MulAssign for ElemExpand<POS> {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl<const POS: u8> DivAssign for ElemExpand<POS> {
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0;
    }
}

impl<const POS: u8> RemAssign for ElemExpand<POS> {
    fn rem_assign(&mut self, rhs: Self) {
        self.0 %= rhs.0;
    }
}

impl<const POS: u8> From<f32> for ElemExpand<POS> {
    fn from(value: f32) -> Self {
        Self::from_f32(value)
    }
}

impl<const POS: u8> From<ElemExpand<POS>> for f32 {
    fn from(val: ElemExpand<POS>) -> Self {
        val.to_f32()
    }
}

impl<const POS: u8> ToPrimitive for ElemExpand<POS> {
    fn to_i64(&self) -> Option<i64> {
        Some((*self).to_f32() as i64)
    }

    fn to_u64(&self) -> Option<u64> {
        Some((*self).to_f32() as u64)
    }

    fn to_f32(&self) -> Option<f32> {
        Some((*self).to_f32())
    }

    fn to_f64(&self) -> Option<f64> {
        Some((*self).to_f32() as f64)
    }
}

impl<const POS: u8> NumCast for ElemExpand<POS> {
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        Some(ElemExpand::from_f32(n.to_f32()?))
    }
}

impl<const POS: u8> CubeType for ElemExpand<POS> {
    type ExpandType = ExpandElementTyped<ElemExpand<POS>>;
}

impl<const POS: u8> CubePrimitive for ElemExpand<POS> {
    /// Return the element type to use on GPU
    fn as_type(scope: &Scope) -> StorageType {
        scope.resolve_type::<Self>().expect("Type to be registered")
    }
}

impl<const POS: u8> From<ElemExpand<POS>> for Variable {
    fn from(val: ElemExpand<POS>) -> Self {
        // TODO: Fix how we create literal.
        Variable::new(
            crate::ir::VariableKind::ConstantScalar(crate::ir::ConstantScalarValue::Float(
                val.0 as f64,
                FloatKind::F32,
            )),
            crate::ir::Type::scalar(ElemType::Float(FloatKind::F32)),
        )
    }
}

impl<const POS: u8> From<ElemExpand<POS>> for ExpandElementTyped<ElemExpand<POS>> {
    fn from(value: ElemExpand<POS>) -> Self {
        let var: Variable = value.into();
        ExpandElementTyped::new(ExpandElement::Plain(var))
    }
}

impl<const POS: u8> IntoRuntime for ElemExpand<POS> {
    fn __expand_runtime_method(self, scope: &mut Scope) -> ExpandElementTyped<Self> {
        let elem: ExpandElementTyped<Self> = ExpandElementTyped::from_lit(scope, self);
        into_runtime_expand_element(scope, elem).into()
    }
}

impl<const POS: u8> Numeric for ElemExpand<POS> {
    fn min_value() -> Self {
        panic!("Can't use min value in comptime with dynamic element type");
    }
    fn max_value() -> Self {
        panic!("Can't use max value in comptime with dynamic element type");
    }
}

impl<const POS: u8> ExpandElementIntoMut for ElemExpand<POS> {
    fn elem_into_mut(scope: &mut Scope, elem: ExpandElement) -> ExpandElement {
        into_mut_expand_element(scope, elem)
    }
}

impl<const POS: u8> Normalize for ElemExpand<POS> {}
impl<const POS: u8> Dot for ElemExpand<POS> {}
impl<const POS: u8> Magnitude for ElemExpand<POS> {}
impl<const POS: u8> Recip for ElemExpand<POS> {}
impl<const POS: u8> Erf for ElemExpand<POS> {}
impl<const POS: u8> Exp for ElemExpand<POS> {}
impl<const POS: u8> Remainder for ElemExpand<POS> {}
impl<const POS: u8> Abs for ElemExpand<POS> {}
impl<const POS: u8> Max for ElemExpand<POS> {}
impl<const POS: u8> Min for ElemExpand<POS> {}
impl<const POS: u8> Clamp for ElemExpand<POS> {}
impl<const POS: u8> Log for ElemExpand<POS> {}
impl<const POS: u8> Log1p for ElemExpand<POS> {}
impl<const POS: u8> Cos for ElemExpand<POS> {}
impl<const POS: u8> Sin for ElemExpand<POS> {}
impl<const POS: u8> Tanh for ElemExpand<POS> {}
impl<const POS: u8> Powf for ElemExpand<POS> {}
impl<const POS: u8, I: CubePrimitive> Powi<I> for ElemExpand<POS> {}
impl<const POS: u8> Sqrt for ElemExpand<POS> {}
impl<const POS: u8> Round for ElemExpand<POS> {}
impl<const POS: u8> Floor for ElemExpand<POS> {}
impl<const POS: u8> Ceil for ElemExpand<POS> {}
impl<const POS: u8> IsNan for ElemExpand<POS> {}
impl<const POS: u8> IsInf for ElemExpand<POS> {}

impl<const POS: u8> Float for ElemExpand<POS> {
    const DIGITS: u32 = 32;

    const EPSILON: Self = ElemExpand::from_f32(half::f16::EPSILON.to_f32_const());

    const INFINITY: Self = ElemExpand::from_f32(f32::INFINITY);

    const MANTISSA_DIGITS: u32 = f32::MANTISSA_DIGITS;

    /// Maximum possible [`tf32`](crate::frontend::tf32) power of 10 exponent
    const MAX_10_EXP: i32 = f32::MAX_10_EXP;
    /// Maximum possible [`tf32`](crate::frontend::tf32) power of 2 exponent
    const MAX_EXP: i32 = f32::MAX_EXP;

    /// Minimum possible normal [`tf32`](crate::frontend::tf32) power of 10 exponent
    const MIN_10_EXP: i32 = f32::MIN_10_EXP;
    /// One greater than the minimum possible normal [`tf32`](crate::frontend::tf32) power of 2 exponent
    const MIN_EXP: i32 = f32::MIN_EXP;

    const MIN_POSITIVE: Self = ElemExpand(f32::MIN_POSITIVE);

    const NAN: Self = ElemExpand::from_f32(f32::NAN);

    const NEG_INFINITY: Self = ElemExpand::from_f32(f32::NEG_INFINITY);

    const RADIX: u32 = 2;

    fn new(val: f32) -> Self {
        ElemExpand::from_f32(val)
    }
}

impl<const POS: u8> Int for ElemExpand<POS> {
    const BITS: u32 = 32;

    fn new(val: i64) -> Self {
        ElemExpand::from_f32(val as f32)
    }
}
impl<const POS: u8> ReverseBits for ElemExpand<POS> {}
impl<const POS: u8> CountOnes for ElemExpand<POS> {}
impl<const POS: u8> BitwiseNot for ElemExpand<POS> {}
impl<const POS: u8> LeadingZeros for ElemExpand<POS> {}
impl<const POS: u8> FindFirstSet for ElemExpand<POS> {}
impl<const POS: u8> SaturatingAdd for ElemExpand<POS> {}
impl<const POS: u8> SaturatingSub for ElemExpand<POS> {}

impl<const POS: u8> BitOr for ElemExpand<POS> {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(BitOr::bitor(self.0 as i32, rhs.0 as i32) as f32)
    }
}
impl<const POS: u8> BitXor for ElemExpand<POS> {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(BitXor::bitxor(self.0 as i32, rhs.0 as i32) as f32)
    }
}

impl<const POS: u8> BitAnd for ElemExpand<POS> {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(BitAnd::bitand(self.0 as i32, rhs.0 as i32) as f32)
    }
}
impl<const POS: u8> BitAndAssign for ElemExpand<POS> {
    fn bitand_assign(&mut self, rhs: Self) {
        let mut value = self.0 as i32;
        BitAndAssign::bitand_assign(&mut value, rhs.0 as i32);
        self.0 = value as f32
    }
}
impl<const POS: u8> BitXorAssign for ElemExpand<POS> {
    fn bitxor_assign(&mut self, rhs: Self) {
        let mut value = self.0 as i32;
        BitXorAssign::bitxor_assign(&mut value, rhs.0 as i32);
        self.0 = value as f32
    }
}

impl<const POS: u8> BitOrAssign for ElemExpand<POS> {
    fn bitor_assign(&mut self, rhs: Self) {
        let mut value = self.0 as i32;
        BitOrAssign::bitor_assign(&mut value, rhs.0 as i32);
        self.0 = value as f32
    }
}
impl<const POS: u8> Shl for ElemExpand<POS> {
    type Output = Self;

    fn shl(self, rhs: Self) -> Self::Output {
        Self(Shl::shl(self.0 as i32, rhs.0 as i32) as f32)
    }
}
impl<const POS: u8> Shr for ElemExpand<POS> {
    type Output = Self;

    fn shr(self, rhs: Self) -> Self::Output {
        Self(Shr::shr(self.0 as i32, rhs.0 as i32) as f32)
    }
}

impl<const POS: u8> ShrAssign<u32> for ElemExpand<POS> {
    fn shr_assign(&mut self, rhs: u32) {
        let mut value = self.0 as i32;
        ShrAssign::shr_assign(&mut value, rhs);
        self.0 = value as f32
    }
}

impl<const POS: u8> ShlAssign<u32> for ElemExpand<POS> {
    fn shl_assign(&mut self, rhs: u32) {
        let mut value = self.0 as i32;
        ShlAssign::shl_assign(&mut value, rhs);
        self.0 = value as f32
    }
}
impl<const POS: u8> Not for ElemExpand<POS> {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self(Not::not(self.0 as i32) as f32)
    }
}

impl<const POS: u8> ScalarArgSettings for ElemExpand<POS> {
    fn register<R: Runtime>(&self, settings: &mut KernelLauncher<R>) {
        settings.register_f32(self.0);
    }
}

impl<const POS: u8> num_traits::Float for ElemExpand<POS> {
    fn nan() -> Self {
        ElemExpand(f32::nan())
    }

    fn infinity() -> Self {
        ElemExpand(f32::infinity())
    }

    fn neg_infinity() -> Self {
        ElemExpand(f32::neg_infinity())
    }

    fn neg_zero() -> Self {
        ElemExpand(f32::neg_zero())
    }

    fn min_value() -> Self {
        ElemExpand(<f32 as num_traits::Float>::min_value())
    }

    fn min_positive_value() -> Self {
        ElemExpand(f32::min_positive_value())
    }

    fn max_value() -> Self {
        ElemExpand(<f32 as num_traits::Float>::max_value())
    }

    fn is_nan(self) -> bool {
        self.0.is_nan()
    }

    fn is_infinite(self) -> bool {
        self.0.is_infinite()
    }

    fn is_finite(self) -> bool {
        self.0.is_finite()
    }

    fn is_normal(self) -> bool {
        self.0.is_normal()
    }

    fn classify(self) -> std::num::FpCategory {
        self.0.classify()
    }

    fn floor(self) -> Self {
        ElemExpand(self.0.floor())
    }

    fn ceil(self) -> Self {
        ElemExpand(self.0.ceil())
    }

    fn round(self) -> Self {
        ElemExpand(self.0.round())
    }

    fn trunc(self) -> Self {
        ElemExpand(self.0.trunc())
    }

    fn fract(self) -> Self {
        ElemExpand(self.0.fract())
    }

    fn abs(self) -> Self {
        ElemExpand(self.0.abs())
    }

    fn signum(self) -> Self {
        ElemExpand(self.0.signum())
    }

    fn is_sign_positive(self) -> bool {
        self.0.is_sign_positive()
    }

    fn is_sign_negative(self) -> bool {
        self.0.is_sign_negative()
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        ElemExpand(self.0.mul_add(a.0, b.0))
    }

    fn recip(self) -> Self {
        ElemExpand(self.0.recip())
    }

    fn powi(self, n: i32) -> Self {
        ElemExpand(self.0.powi(n))
    }

    fn powf(self, n: Self) -> Self {
        ElemExpand(self.0.powf(n.0))
    }

    fn sqrt(self) -> Self {
        ElemExpand(self.0.sqrt())
    }

    fn exp(self) -> Self {
        ElemExpand(self.0.exp())
    }

    fn exp2(self) -> Self {
        ElemExpand(self.0.exp2())
    }

    fn ln(self) -> Self {
        ElemExpand(self.0.ln())
    }

    fn log(self, base: Self) -> Self {
        ElemExpand(self.0.log(base.0))
    }

    fn log2(self) -> Self {
        ElemExpand(self.0.log2())
    }

    fn log10(self) -> Self {
        ElemExpand(self.0.log10())
    }

    fn max(self, other: Self) -> Self {
        ElemExpand(self.0.max(other.0))
    }

    fn min(self, other: Self) -> Self {
        ElemExpand(self.0.min(other.0))
    }

    fn abs_sub(self, other: Self) -> Self {
        ElemExpand((self.0 - other.0).abs())
    }

    fn cbrt(self) -> Self {
        ElemExpand(self.0.cbrt())
    }

    fn hypot(self, other: Self) -> Self {
        ElemExpand(self.0.hypot(other.0))
    }

    fn sin(self) -> Self {
        ElemExpand(self.0.sin())
    }

    fn cos(self) -> Self {
        ElemExpand(self.0.cos())
    }

    fn tan(self) -> Self {
        ElemExpand(self.0.tan())
    }

    fn asin(self) -> Self {
        ElemExpand(self.0.asin())
    }

    fn acos(self) -> Self {
        ElemExpand(self.0.acos())
    }

    fn atan(self) -> Self {
        ElemExpand(self.0.atan())
    }

    fn atan2(self, other: Self) -> Self {
        ElemExpand(self.0.atan2(other.0))
    }

    fn sin_cos(self) -> (Self, Self) {
        let (a, b) = self.0.sin_cos();
        (ElemExpand(a), ElemExpand(b))
    }

    fn exp_m1(self) -> Self {
        ElemExpand(self.0.exp_m1())
    }

    fn ln_1p(self) -> Self {
        ElemExpand(self.0.ln_1p())
    }

    fn sinh(self) -> Self {
        ElemExpand(self.0.sinh())
    }

    fn cosh(self) -> Self {
        ElemExpand(self.0.cosh())
    }

    fn tanh(self) -> Self {
        ElemExpand(self.0.tanh())
    }

    fn asinh(self) -> Self {
        ElemExpand(self.0.asinh())
    }

    fn acosh(self) -> Self {
        ElemExpand(self.0.acosh())
    }

    fn atanh(self) -> Self {
        ElemExpand(self.0.atanh())
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        self.0.integer_decode()
    }
}

impl<const POS: u8> Num for ElemExpand<POS> {
    type FromStrRadixErr = <f32 as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Ok(ElemExpand(f32::from_str_radix(str, radix)?))
    }
}

impl<const POS: u8> One for ElemExpand<POS> {
    fn one() -> Self {
        ElemExpand(1.0)
    }
}

impl<const POS: u8> Zero for ElemExpand<POS> {
    fn zero() -> Self {
        ElemExpand(0.0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0.0
    }
}
