use core::f32;
use core::{
    cmp::Ordering,
    ops::{Div, DivAssign, Mul, MulAssign, Rem, RemAssign},
};

use bytemuck::{Pod, Zeroable};
use derive_more::derive::{
    Add, AddAssign, Display, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
use num_traits::{Num, NumCast, One, ToPrimitive, Zero};

/// A floating point type with relaxed precision, minimum [`f16`], max [`f32`].
///
#[allow(non_camel_case_types)]
#[repr(transparent)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(
    Clone,
    Copy,
    Default,
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
pub struct flex32(f32);

impl flex32 {
    /// Minimum positive flex32 value
    pub const MIN_POSITIVE: Self = Self(half::f16::MIN_POSITIVE.to_f32_const());

    /// Create a `flex32` from [`prim@f32`]
    pub const fn from_f32(val: f32) -> Self {
        flex32(val)
    }

    /// Create a `flex32` from [`prim@f64`]
    pub const fn from_f64(val: f64) -> Self {
        flex32(val as f32)
    }

    /// Turn a `flex32` into [`prim@f32`]
    pub const fn to_f32(self) -> f32 {
        self.0
    }

    /// Turn a `flex32` into [`prim@f64`]
    pub const fn to_f64(self) -> f64 {
        self.0 as f64
    }

    /// Compare two flex32 numbers
    pub fn total_cmp(&self, other: &flex32) -> Ordering {
        self.0.total_cmp(&other.0)
    }

    /// Whether this flex32 represents `NaN`
    pub fn is_nan(&self) -> bool {
        self.0.is_nan()
    }
}

impl Mul for flex32 {
    type Output = flex32;

    fn mul(self, rhs: Self) -> Self::Output {
        flex32(self.0 * rhs.0)
    }
}

impl Div for flex32 {
    type Output = flex32;

    fn div(self, rhs: Self) -> Self::Output {
        flex32(self.0 / rhs.0)
    }
}

impl Rem for flex32 {
    type Output = flex32;

    fn rem(self, rhs: Self) -> Self::Output {
        flex32(self.0 % rhs.0)
    }
}

impl MulAssign for flex32 {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl DivAssign for flex32 {
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0;
    }
}

impl RemAssign for flex32 {
    fn rem_assign(&mut self, rhs: Self) {
        self.0 %= rhs.0;
    }
}

impl From<f32> for flex32 {
    fn from(value: f32) -> Self {
        Self::from_f32(value)
    }
}

impl From<flex32> for f32 {
    fn from(val: flex32) -> Self {
        val.to_f32()
    }
}

impl ToPrimitive for flex32 {
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

impl NumCast for flex32 {
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        Some(flex32::from_f32(n.to_f32()?))
    }
}

impl num_traits::Float for flex32 {
    fn nan() -> Self {
        flex32(f32::nan())
    }

    fn infinity() -> Self {
        flex32(f32::infinity())
    }

    fn neg_infinity() -> Self {
        flex32(f32::neg_infinity())
    }

    fn neg_zero() -> Self {
        flex32(f32::neg_zero())
    }

    fn min_value() -> Self {
        flex32(<f32 as num_traits::Float>::min_value())
    }

    fn min_positive_value() -> Self {
        flex32(f32::min_positive_value())
    }

    fn max_value() -> Self {
        flex32(<f32 as num_traits::Float>::max_value())
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
        flex32(self.0.floor())
    }

    fn ceil(self) -> Self {
        flex32(self.0.ceil())
    }

    fn round(self) -> Self {
        flex32(self.0.round())
    }

    fn trunc(self) -> Self {
        flex32(self.0.trunc())
    }

    fn fract(self) -> Self {
        flex32(self.0.fract())
    }

    fn abs(self) -> Self {
        flex32(self.0.abs())
    }

    fn signum(self) -> Self {
        flex32(self.0.signum())
    }

    fn is_sign_positive(self) -> bool {
        self.0.is_sign_positive()
    }

    fn is_sign_negative(self) -> bool {
        self.0.is_sign_negative()
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        flex32(self.0.mul_add(a.0, b.0))
    }

    fn recip(self) -> Self {
        flex32(self.0.recip())
    }

    fn powi(self, n: i32) -> Self {
        flex32(self.0.powi(n))
    }

    fn powf(self, n: Self) -> Self {
        flex32(self.0.powf(n.0))
    }

    fn sqrt(self) -> Self {
        flex32(self.0.sqrt())
    }

    fn exp(self) -> Self {
        flex32(self.0.exp())
    }

    fn exp2(self) -> Self {
        flex32(self.0.exp2())
    }

    fn ln(self) -> Self {
        flex32(self.0.ln())
    }

    fn log(self, base: Self) -> Self {
        flex32(self.0.log(base.0))
    }

    fn log2(self) -> Self {
        flex32(self.0.log2())
    }

    fn log10(self) -> Self {
        flex32(self.0.log10())
    }

    fn max(self, other: Self) -> Self {
        flex32(self.0.max(other.0))
    }

    fn min(self, other: Self) -> Self {
        flex32(self.0.min(other.0))
    }

    fn abs_sub(self, other: Self) -> Self {
        flex32((self.0 - other.0).abs())
    }

    fn cbrt(self) -> Self {
        flex32(self.0.cbrt())
    }

    fn hypot(self, other: Self) -> Self {
        flex32(self.0.hypot(other.0))
    }

    fn sin(self) -> Self {
        flex32(self.0.sin())
    }

    fn cos(self) -> Self {
        flex32(self.0.cos())
    }

    fn tan(self) -> Self {
        flex32(self.0.tan())
    }

    fn asin(self) -> Self {
        flex32(self.0.asin())
    }

    fn acos(self) -> Self {
        flex32(self.0.acos())
    }

    fn atan(self) -> Self {
        flex32(self.0.atan())
    }

    fn atan2(self, other: Self) -> Self {
        flex32(self.0.atan2(other.0))
    }

    fn sin_cos(self) -> (Self, Self) {
        let (a, b) = self.0.sin_cos();
        (flex32(a), flex32(b))
    }

    fn exp_m1(self) -> Self {
        flex32(self.0.exp_m1())
    }

    fn ln_1p(self) -> Self {
        flex32(self.0.ln_1p())
    }

    fn sinh(self) -> Self {
        flex32(self.0.sinh())
    }

    fn cosh(self) -> Self {
        flex32(self.0.cosh())
    }

    fn tanh(self) -> Self {
        flex32(self.0.tanh())
    }

    fn asinh(self) -> Self {
        flex32(self.0.asinh())
    }

    fn acosh(self) -> Self {
        flex32(self.0.acosh())
    }

    fn atanh(self) -> Self {
        flex32(self.0.atanh())
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        self.0.integer_decode()
    }
}

impl Num for flex32 {
    type FromStrRadixErr = <f32 as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Ok(flex32(f32::from_str_radix(str, radix)?))
    }
}

impl One for flex32 {
    fn one() -> Self {
        flex32(1.0)
    }
}

impl Zero for flex32 {
    fn zero() -> Self {
        flex32(0.0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0.0
    }
}
