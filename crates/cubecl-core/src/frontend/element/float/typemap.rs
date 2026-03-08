//! This module contains a configurable [element type](ElemExpand) for floats to be used during
//! kernel expansion to speed up Rust compilation.
//!
//! Expand functions don't need to be generated for different element types even if they are generic
//! over one, since the only use of numeric element types is to map to the [elem IR enum](Elem).
//!
//! This can be done dynamically using the scope instead, reducing the binary size and the
//! compilation time of kernels significantly.
//!
//! You can still have multiple element types in a single kernel, since [`ElemExpand`] uses const
//! generics to differentiate between float kinds.

use core::f32;
use core::{
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
use float_ord::FloatOrd;
use num_traits::{Num, NumCast, One, ToPrimitive, Zero};
use serde::Serialize;

use crate::{
    ir::{FloatKind, Scope, Variable},
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
pub struct ElemExpand<const POS: usize>(f32);

/// A fake float element type that can be configured to map to any other element type.
pub type FloatExpand<const POS: usize> = ElemExpand<POS>;

/// A fake numeric element type that can be configured to map to any other element type.
pub type NumericExpand<const POS: usize> = ElemExpand<POS>;

/// A fake constant type that can be configured to map to any comptime value.
#[derive(Clone, Copy, Debug)]
pub struct SizeExpand<const POS: usize>;

impl<const POS: usize> ElemExpand<POS> {
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

impl<const POS: usize> Mul for ElemExpand<POS> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        ElemExpand(self.0 * rhs.0)
    }
}

impl<const POS: usize> Div for ElemExpand<POS> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        ElemExpand(self.0 / rhs.0)
    }
}

impl<const POS: usize> Rem for ElemExpand<POS> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        ElemExpand(self.0 % rhs.0)
    }
}

impl<const POS: usize> MulAssign for ElemExpand<POS> {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl<const POS: usize> DivAssign for ElemExpand<POS> {
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0;
    }
}

impl<const POS: usize> RemAssign for ElemExpand<POS> {
    fn rem_assign(&mut self, rhs: Self) {
        self.0 %= rhs.0;
    }
}

impl<const POS: usize> From<f32> for ElemExpand<POS> {
    fn from(value: f32) -> Self {
        Self::from_f32(value)
    }
}

impl<const POS: usize> From<ElemExpand<POS>> for f32 {
    fn from(val: ElemExpand<POS>) -> Self {
        val.to_f32()
    }
}

impl<const POS: usize> ToPrimitive for ElemExpand<POS> {
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

impl<const POS: usize> NumCast for ElemExpand<POS> {
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        Some(ElemExpand::from_f32(n.to_f32()?))
    }
}

impl<const POS: usize> CubeType for ElemExpand<POS> {
    type ExpandType = ExpandElementTyped<ElemExpand<POS>>;
}

impl<const POS: usize> CubePrimitive for ElemExpand<POS> {
    /// Return the element type to use on GPU
    fn as_type(scope: &Scope) -> Type {
        scope.resolve_type::<Self>().expect("Type to be registered")
    }

    fn from_const_value(_value: ConstantValue) -> Self {
        unimplemented!("Can't turn `ElemExpand` into a constant value")
    }
}

impl<const POS: usize> From<ElemExpand<POS>> for ConstantValue {
    fn from(val: ElemExpand<POS>) -> Self {
        val.0.into()
    }
}

impl<const POS: usize> From<ElemExpand<POS>> for Variable {
    fn from(val: ElemExpand<POS>) -> Self {
        // TODO: Fix how we create literal.
        Variable::constant(val.0.into(), FloatKind::F32)
    }
}

impl<const POS: usize> From<ElemExpand<POS>> for ExpandElementTyped<ElemExpand<POS>> {
    fn from(value: ElemExpand<POS>) -> Self {
        let var: Variable = value.into();
        ExpandElementTyped::new(ExpandElement::Plain(var))
    }
}

impl<const POS: usize> IntoRuntime for ElemExpand<POS> {
    fn __expand_runtime_method(self, scope: &mut Scope) -> ExpandElementTyped<Self> {
        let elem: ExpandElementTyped<Self> = ExpandElementTyped::from_lit(scope, self);
        into_runtime_expand_element(scope, elem).into()
    }
}

impl<const POS: usize> Numeric for ElemExpand<POS> {
    fn min_value() -> Self {
        panic!("Can't use min value in comptime with dynamic element type");
    }
    fn max_value() -> Self {
        panic!("Can't use max value in comptime with dynamic element type");
    }
}

impl<const POS: usize> ExpandElementAssign for ElemExpand<POS> {}

impl<const POS: usize> ScalarArgSettings for ElemExpand<POS> {
    fn register<R: Runtime>(&self, _launcher: &mut KernelLauncher<R>) {
        panic!("Can't launch `ElemExpand` as scalar")
    }
}

impl<const POS: usize> Normalize for ElemExpand<POS> {}
impl<const POS: usize> Dot for ElemExpand<POS> {}
impl<const POS: usize> Magnitude for ElemExpand<POS> {}
impl<const POS: usize> Recip for ElemExpand<POS> {}
impl<const POS: usize> Erf for ElemExpand<POS> {}
impl<const POS: usize> Exp for ElemExpand<POS> {}
impl<const POS: usize> Remainder for ElemExpand<POS> {}
impl<const POS: usize> Abs for ElemExpand<POS> {}
impl<const POS: usize> Log for ElemExpand<POS> {}
impl<const POS: usize> Log1p for ElemExpand<POS> {}
impl<const POS: usize> Cos for ElemExpand<POS> {}
impl<const POS: usize> Sin for ElemExpand<POS> {}
impl<const POS: usize> Tan for ElemExpand<POS> {}
impl<const POS: usize> Tanh for ElemExpand<POS> {}
impl<const POS: usize> Sinh for ElemExpand<POS> {}
impl<const POS: usize> Cosh for ElemExpand<POS> {}
impl<const POS: usize> ArcCos for ElemExpand<POS> {}
impl<const POS: usize> ArcSin for ElemExpand<POS> {}
impl<const POS: usize> ArcTan for ElemExpand<POS> {}
impl<const POS: usize> ArcSinh for ElemExpand<POS> {}
impl<const POS: usize> ArcCosh for ElemExpand<POS> {}
impl<const POS: usize> ArcTanh for ElemExpand<POS> {}
impl<const POS: usize> Degrees for ElemExpand<POS> {}
impl<const POS: usize> Radians for ElemExpand<POS> {}
impl<const POS: usize> ArcTan2 for ElemExpand<POS> {}
impl<const POS: usize> Powf for ElemExpand<POS> {}
impl<const POS: usize, I: CubePrimitive> Powi<I> for ElemExpand<POS> {}
impl<const POS: usize> Hypot for ElemExpand<POS> {}
impl<const POS: usize> Rhypot for ElemExpand<POS> {}
impl<const POS: usize> Sqrt for ElemExpand<POS> {}
impl<const POS: usize> InverseSqrt for ElemExpand<POS> {}
impl<const POS: usize> Round for ElemExpand<POS> {}
impl<const POS: usize> Floor for ElemExpand<POS> {}
impl<const POS: usize> Ceil for ElemExpand<POS> {}
impl<const POS: usize> Trunc for ElemExpand<POS> {}
impl<const POS: usize> IsNan for ElemExpand<POS> {}
impl<const POS: usize> IsInf for ElemExpand<POS> {}

#[allow(clippy::non_canonical_partial_ord_impl)]
impl<const POS: usize> PartialOrd for ElemExpand<POS> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        FloatOrd(self.0).partial_cmp(&FloatOrd(other.0))
    }
}

impl<const POS: usize> Ord for ElemExpand<POS> {
    fn cmp(&self, other: &Self) -> Ordering {
        FloatOrd(self.0).cmp(&FloatOrd(other.0))
    }
}

impl<const POS: usize> Float for ElemExpand<POS> {
    const DIGITS: u32 = 32;

    const EPSILON: Self = ElemExpand::from_f32(half::f16::EPSILON.to_f32_const());

    const INFINITY: Self = ElemExpand::from_f32(f32::INFINITY);

    const MANTISSA_DIGITS: u32 = f32::MANTISSA_DIGITS;

    /// Maximum possible [`cubecl_common::tf32`] power of 10 exponent
    const MAX_10_EXP: i32 = f32::MAX_10_EXP;
    /// Maximum possible [`cubecl_common::tf32`] power of 2 exponent
    const MAX_EXP: i32 = f32::MAX_EXP;

    /// Minimum possible normal [`cubecl_common::tf32`] power of 10 exponent
    const MIN_10_EXP: i32 = f32::MIN_10_EXP;

    /// One greater than the minimum possible normal [`cubecl_common::tf32`] power of 2 exponent
    const MIN_EXP: i32 = f32::MIN_EXP;

    const MIN_POSITIVE: Self = ElemExpand(f32::MIN_POSITIVE);

    const NAN: Self = ElemExpand::from_f32(f32::NAN);

    const NEG_INFINITY: Self = ElemExpand::from_f32(f32::NEG_INFINITY);

    const RADIX: u32 = 2;

    fn new(val: f32) -> Self {
        ElemExpand::from_f32(val)
    }
}

impl<const POS: usize> Int for ElemExpand<POS> {
    const BITS: u32 = 32;

    fn new(val: i64) -> Self {
        ElemExpand::from_f32(val as f32)
    }
}
impl<const POS: usize> CubeNot for ElemExpand<POS> {}
impl<const POS: usize> ReverseBits for ElemExpand<POS> {}
impl<const POS: usize> CountOnes for ElemExpand<POS> {}
impl<const POS: usize> LeadingZeros for ElemExpand<POS> {}
impl<const POS: usize> TrailingZeros for ElemExpand<POS> {}
impl<const POS: usize> FindFirstSet for ElemExpand<POS> {}
impl<const POS: usize> SaturatingAdd for ElemExpand<POS> {}
impl<const POS: usize> SaturatingSub for ElemExpand<POS> {}
impl<const POS: usize> Eq for ElemExpand<POS> {}
impl<const POS: usize> core::hash::Hash for ElemExpand<POS> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl<const POS: usize> Not for ElemExpand<POS> {
    type Output = Self;

    fn not(self) -> Self::Output {
        ElemExpand(!(self.0 as i64) as f32)
    }
}

impl<const POS: usize> BitOr for ElemExpand<POS> {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(BitOr::bitor(self.0 as i32, rhs.0 as i32) as f32)
    }
}
impl<const POS: usize> BitXor for ElemExpand<POS> {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(BitXor::bitxor(self.0 as i32, rhs.0 as i32) as f32)
    }
}

impl<const POS: usize> BitAnd for ElemExpand<POS> {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(BitAnd::bitand(self.0 as i32, rhs.0 as i32) as f32)
    }
}
impl<const POS: usize> BitAndAssign for ElemExpand<POS> {
    fn bitand_assign(&mut self, rhs: Self) {
        let mut value = self.0 as i32;
        BitAndAssign::bitand_assign(&mut value, rhs.0 as i32);
        self.0 = value as f32
    }
}
impl<const POS: usize> BitXorAssign for ElemExpand<POS> {
    fn bitxor_assign(&mut self, rhs: Self) {
        let mut value = self.0 as i32;
        BitXorAssign::bitxor_assign(&mut value, rhs.0 as i32);
        self.0 = value as f32
    }
}

impl<const POS: usize> BitOrAssign for ElemExpand<POS> {
    fn bitor_assign(&mut self, rhs: Self) {
        let mut value = self.0 as i32;
        BitOrAssign::bitor_assign(&mut value, rhs.0 as i32);
        self.0 = value as f32
    }
}
impl<const POS: usize> Shl for ElemExpand<POS> {
    type Output = Self;

    fn shl(self, rhs: Self) -> Self::Output {
        Self(Shl::shl(self.0 as i32, rhs.0 as i32) as f32)
    }
}
impl<const POS: usize> Shr for ElemExpand<POS> {
    type Output = Self;

    fn shr(self, rhs: Self) -> Self::Output {
        Self(Shr::shr(self.0 as i32, rhs.0 as i32) as f32)
    }
}

impl<const POS: usize> ShrAssign<u32> for ElemExpand<POS> {
    fn shr_assign(&mut self, rhs: u32) {
        let mut value = self.0 as i32;
        ShrAssign::shr_assign(&mut value, rhs);
        self.0 = value as f32
    }
}

impl<const POS: usize> ShlAssign<u32> for ElemExpand<POS> {
    fn shl_assign(&mut self, rhs: u32) {
        let mut value = self.0 as i32;
        ShlAssign::shl_assign(&mut value, rhs);
        self.0 = value as f32
    }
}

impl<const POS: usize> num_traits::Float for ElemExpand<POS> {
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

    fn classify(self) -> core::num::FpCategory {
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

impl<const POS: usize> Num for ElemExpand<POS> {
    type FromStrRadixErr = <f32 as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Ok(ElemExpand(f32::from_str_radix(str, radix)?))
    }
}

impl<const POS: usize> One for ElemExpand<POS> {
    fn one() -> Self {
        ElemExpand(1.0)
    }
}

impl<const POS: usize> Zero for ElemExpand<POS> {
    fn zero() -> Self {
        ElemExpand(0.0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0.0
    }
}
