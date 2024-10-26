use core::f32;
use std::{
    cmp::Ordering,
    num::NonZero,
    ops::{Div, DivAssign, Mul, MulAssign, Rem, RemAssign},
};

use bytemuck::{Pod, Zeroable};
use derive_more::derive::{
    Add, AddAssign, Display, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
use num_traits::{Num, NumCast, One, ToPrimitive, Zero};
use serde::Serialize;

use crate::{
    ir::{Elem, FloatKind, Item},
    prelude::Numeric,
    unexpanded,
};

use super::{
    init_expand_element, CubeContext, CubePrimitive, CubeType, ExpandElement,
    ExpandElementBaseInit, ExpandElementTyped, Float, Init, IntoRuntime, KernelBuilder,
    KernelLauncher, LaunchArgExpand, Runtime, ScalarArgSettings, Vectorized,
};

/// A floating point type with relaxed precision, minimum [`f16`], max [`f32`].
///
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
pub struct minf16(f32);

impl minf16 {
    pub const fn from_f32(val: f32) -> Self {
        minf16(val)
    }

    pub const fn from_f64(val: f64) -> Self {
        minf16(val as f32)
    }

    pub const fn to_f32(self) -> f32 {
        self.0
    }

    pub const fn to_f64(self) -> f64 {
        self.0 as f64
    }

    pub fn total_cmp(&self, other: &minf16) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl Mul for minf16 {
    type Output = minf16;

    fn mul(self, rhs: Self) -> Self::Output {
        minf16(self.0 * rhs.0)
    }
}

impl Div for minf16 {
    type Output = minf16;

    fn div(self, rhs: Self) -> Self::Output {
        minf16(self.0 / rhs.0)
    }
}

impl Rem for minf16 {
    type Output = minf16;

    fn rem(self, rhs: Self) -> Self::Output {
        minf16(self.0 % rhs.0)
    }
}

impl MulAssign for minf16 {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl DivAssign for minf16 {
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0;
    }
}

impl RemAssign for minf16 {
    fn rem_assign(&mut self, rhs: Self) {
        self.0 %= rhs.0;
    }
}

impl From<f32> for minf16 {
    fn from(value: f32) -> Self {
        Self::from_f32(value)
    }
}

impl From<minf16> for f32 {
    fn from(val: minf16) -> Self {
        val.to_f32()
    }
}

impl ToPrimitive for minf16 {
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

impl NumCast for minf16 {
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        Some(minf16::from_f32(n.to_f32()?))
    }
}

impl CubeType for minf16 {
    type ExpandType = ExpandElementTyped<minf16>;
}

impl CubePrimitive for minf16 {
    /// Return the element type to use on GPU
    fn as_elem() -> Elem {
        Elem::Float(FloatKind::Relaxed)
    }
}

impl IntoRuntime for minf16 {
    fn __expand_runtime_method(self, context: &mut CubeContext) -> ExpandElementTyped<Self> {
        let expand: ExpandElementTyped<Self> = self.into();
        Init::init(expand, context)
    }
}

impl Numeric for minf16 {
    const MAX: Self = minf16::from_f32(f32::MAX);
    const MIN: Self = minf16::from_f32(f32::MIN);
}

impl Vectorized for minf16 {
    fn vectorization_factor(&self) -> u32 {
        1
    }

    fn vectorize(self, _factor: u32) -> Self {
        unexpanded!()
    }
}

impl ExpandElementBaseInit for minf16 {
    fn init_elem(context: &mut CubeContext, elem: ExpandElement) -> ExpandElement {
        init_expand_element(context, elem)
    }
}

impl Float for minf16 {
    const DIGITS: u32 = 32;

    const EPSILON: Self = minf16::from_f32(half::f16::EPSILON.to_f32_const());

    const INFINITY: Self = minf16::from_f32(f32::INFINITY);

    const MANTISSA_DIGITS: u32 = f32::MANTISSA_DIGITS;

    /// Maximum possible [`tf32`] power of 10 exponent
    const MAX_10_EXP: i32 = f32::MAX_10_EXP;
    /// Maximum possible [`tf32`] power of 2 exponent
    const MAX_EXP: i32 = f32::MAX_EXP;

    /// Minimum possible normal [`tf32`] power of 10 exponent
    const MIN_10_EXP: i32 = f32::MIN_10_EXP;
    /// One greater than the minimum possible normal [`v`] power of 2 exponent
    const MIN_EXP: i32 = f32::MIN_EXP;

    const MIN_POSITIVE: Self = minf16(f32::MIN_POSITIVE);

    const NAN: Self = minf16::from_f32(f32::NAN);

    const NEG_INFINITY: Self = minf16::from_f32(f32::NEG_INFINITY);

    const RADIX: u32 = 2;

    fn new(val: f32) -> Self {
        minf16::from_f32(val)
    }

    fn vectorized(_val: f32, _vectorization: u32) -> Self {
        unexpanded!()
    }

    fn vectorized_empty(_vectorization: u32) -> Self {
        unexpanded!()
    }

    fn __expand_vectorized_empty(
        context: &mut super::CubeContext,
        vectorization: u32,
    ) -> <Self as super::CubeType>::ExpandType {
        context
            .create_local_variable(Item::vectorized(
                Self::as_elem(),
                NonZero::new(vectorization as u8),
            ))
            .into()
    }
}

impl LaunchArgExpand for minf16 {
    type CompilationArg = ();

    fn expand(_: &Self::CompilationArg, builder: &mut KernelBuilder) -> ExpandElementTyped<Self> {
        builder.scalar(minf16::as_elem()).into()
    }
}

impl ScalarArgSettings for minf16 {
    fn register<R: Runtime>(&self, settings: &mut KernelLauncher<R>) {
        settings.register_f32(self.0);
    }
}

impl num_traits::Float for minf16 {
    fn nan() -> Self {
        minf16(f32::nan())
    }

    fn infinity() -> Self {
        minf16(f32::infinity())
    }

    fn neg_infinity() -> Self {
        minf16(f32::neg_infinity())
    }

    fn neg_zero() -> Self {
        minf16(f32::neg_zero())
    }

    fn min_value() -> Self {
        minf16(f32::min_value())
    }

    fn min_positive_value() -> Self {
        minf16(f32::min_positive_value())
    }

    fn max_value() -> Self {
        minf16(f32::max_value())
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
        minf16(self.0.floor())
    }

    fn ceil(self) -> Self {
        minf16(self.0.ceil())
    }

    fn round(self) -> Self {
        minf16(self.0.round())
    }

    fn trunc(self) -> Self {
        minf16(self.0.trunc())
    }

    fn fract(self) -> Self {
        minf16(self.0.fract())
    }

    fn abs(self) -> Self {
        minf16(self.0.abs())
    }

    fn signum(self) -> Self {
        minf16(self.0.signum())
    }

    fn is_sign_positive(self) -> bool {
        self.0.is_sign_positive()
    }

    fn is_sign_negative(self) -> bool {
        self.0.is_sign_negative()
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        minf16(self.0.mul_add(a.0, b.0))
    }

    fn recip(self) -> Self {
        minf16(self.0.recip())
    }

    fn powi(self, n: i32) -> Self {
        minf16(self.0.powi(n))
    }

    fn powf(self, n: Self) -> Self {
        minf16(self.0.powf(n.0))
    }

    fn sqrt(self) -> Self {
        minf16(self.0.sqrt())
    }

    fn exp(self) -> Self {
        minf16(self.0.exp())
    }

    fn exp2(self) -> Self {
        minf16(self.0.exp2())
    }

    fn ln(self) -> Self {
        minf16(self.0.ln())
    }

    fn log(self, base: Self) -> Self {
        minf16(self.0.log(base.0))
    }

    fn log2(self) -> Self {
        minf16(self.0.log2())
    }

    fn log10(self) -> Self {
        minf16(self.0.log10())
    }

    fn max(self, other: Self) -> Self {
        minf16(self.0.max(other.0))
    }

    fn min(self, other: Self) -> Self {
        minf16(self.0.min(other.0))
    }

    fn abs_sub(self, other: Self) -> Self {
        minf16((self.0 - other.0).abs())
    }

    fn cbrt(self) -> Self {
        minf16(self.0.cbrt())
    }

    fn hypot(self, other: Self) -> Self {
        minf16(self.0.hypot(other.0))
    }

    fn sin(self) -> Self {
        minf16(self.0.sin())
    }

    fn cos(self) -> Self {
        minf16(self.0.cos())
    }

    fn tan(self) -> Self {
        minf16(self.0.tan())
    }

    fn asin(self) -> Self {
        minf16(self.0.asin())
    }

    fn acos(self) -> Self {
        minf16(self.0.acos())
    }

    fn atan(self) -> Self {
        minf16(self.0.atan())
    }

    fn atan2(self, other: Self) -> Self {
        minf16(self.0.atan2(other.0))
    }

    fn sin_cos(self) -> (Self, Self) {
        let (a, b) = self.0.sin_cos();
        (minf16(a), minf16(b))
    }

    fn exp_m1(self) -> Self {
        minf16(self.0.exp_m1())
    }

    fn ln_1p(self) -> Self {
        minf16(self.0.ln_1p())
    }

    fn sinh(self) -> Self {
        minf16(self.0.sinh())
    }

    fn cosh(self) -> Self {
        minf16(self.0.cosh())
    }

    fn tanh(self) -> Self {
        minf16(self.0.tanh())
    }

    fn asinh(self) -> Self {
        minf16(self.0.asinh())
    }

    fn acosh(self) -> Self {
        minf16(self.0.acosh())
    }

    fn atanh(self) -> Self {
        minf16(self.0.atanh())
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        self.0.integer_decode()
    }
}

impl Num for minf16 {
    type FromStrRadixErr = <f32 as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Ok(minf16(f32::from_str_radix(str, radix)?))
    }
}

impl One for minf16 {
    fn one() -> Self {
        minf16(1.0)
    }
}

impl Zero for minf16 {
    fn zero() -> Self {
        minf16(0.0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0.0
    }
}
