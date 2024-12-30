use core::f32;
use std::{
    cmp::Ordering,
    ops::{Div, DivAssign, Mul, MulAssign, Rem, RemAssign},
};

use bytemuck::{Pod, Zeroable};
use derive_more::derive::{
    Add, AddAssign, Display, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
use num_traits::{Num, NumCast, One, ToPrimitive, Zero};
use serde::Serialize;

use crate::{
    ir::{Elem, FloatKind, Variable},
    prelude::Numeric,
};

use super::{
    init_expand_element, Abs, Ceil, Clamp, Cos, CubeContext, CubeIndex, CubeIndexMut,
    CubePrimitive, CubeType, Dot, Erf, Exp, ExpandElement, ExpandElementBaseInit,
    ExpandElementTyped, Float, Floor, Index, Init, IntoRuntime, KernelBuilder, KernelLauncher,
    LaunchArgExpand, Log, Log1p, Magnitude, Max, Min, Normalize, Powf, Recip, Remainder, Round,
    Runtime, ScalarArgSettings, Sin, Sqrt, Tanh,
};

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
pub struct FloatExpand<const POS: u8>(f32);
pub type NumericExpand<const POS: u8> = FloatExpand<POS>;
pub type IntExpand<const POS: u8> = FloatExpand<POS>;

impl<const POS: u8> FloatExpand<POS> {
    pub const MIN_POSITIVE: Self = Self(half::f16::MIN_POSITIVE.to_f32_const());

    pub const fn from_f32(val: f32) -> Self {
        FloatExpand(val)
    }

    pub const fn from_f64(val: f64) -> Self {
        FloatExpand(val as f32)
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

impl<const POS: u8> Mul for FloatExpand<POS> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        FloatExpand(self.0 * rhs.0)
    }
}

impl<const POS: u8> Div for FloatExpand<POS> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        FloatExpand(self.0 / rhs.0)
    }
}

impl<const POS: u8> Rem for FloatExpand<POS> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        FloatExpand(self.0 % rhs.0)
    }
}

impl<const POS: u8> MulAssign for FloatExpand<POS> {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl<const POS: u8> DivAssign for FloatExpand<POS> {
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0;
    }
}

impl<const POS: u8> RemAssign for FloatExpand<POS> {
    fn rem_assign(&mut self, rhs: Self) {
        self.0 %= rhs.0;
    }
}

impl<const POS: u8> From<f32> for FloatExpand<POS> {
    fn from(value: f32) -> Self {
        Self::from_f32(value)
    }
}

impl<const POS: u8> From<FloatExpand<POS>> for f32 {
    fn from(val: FloatExpand<POS>) -> Self {
        val.to_f32()
    }
}

impl<const POS: u8> ToPrimitive for FloatExpand<POS> {
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

impl<const POS: u8> NumCast for FloatExpand<POS> {
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        Some(FloatExpand::from_f32(n.to_f32()?))
    }
}

impl<const POS: u8> CubeType for FloatExpand<POS> {
    type ExpandType = ExpandElementTyped<FloatExpand<POS>>;
}

impl<const POS: u8> CubePrimitive for FloatExpand<POS> {
    /// Return the element type to use on GPU
    fn as_elem(context: &CubeContext) -> Elem {
        context
            .resolve_elem::<Self>()
            .expect("Type to be registered")
    }
}

impl<const POS: u8> Into<Variable> for FloatExpand<POS> {
    fn into(self) -> Variable {
        // TODO: Fix how we create literal.
        Variable::new(
            crate::ir::VariableKind::ConstantScalar(crate::ir::ConstantScalarValue::Float(
                self.0 as f64,
                FloatKind::F32,
            )),
            crate::ir::Item::new(Elem::Float(FloatKind::F32)),
        )
    }
}

impl<const POS: u8> Into<ExpandElementTyped<Self>> for FloatExpand<POS> {
    fn into(self) -> ExpandElementTyped<Self> {
        let var: Variable = self.into();
        ExpandElementTyped::new(ExpandElement::Plain(var))
    }
}

impl<const POS: u8> IntoRuntime for FloatExpand<POS> {
    fn __expand_runtime_method(self, context: &mut CubeContext) -> ExpandElementTyped<Self> {
        let expand: ExpandElementTyped<Self> = ExpandElementTyped::from_lit(context, self);
        Init::init(expand, context)
    }
}

impl<const POS: u8> Numeric for FloatExpand<POS> {
    fn min_value() -> Self {
        <Self as num_traits::Float>::min_value()
    }
    fn max_value() -> Self {
        <Self as num_traits::Float>::min_value()
    }
}

impl<const POS: u8> ExpandElementBaseInit for FloatExpand<POS> {
    fn init_elem(context: &mut CubeContext, elem: ExpandElement) -> ExpandElement {
        init_expand_element(context, elem)
    }
}

impl<const POS: u8> Normalize for FloatExpand<POS> {}
impl<const POS: u8> Dot for FloatExpand<POS> {}
impl<const POS: u8> Magnitude for FloatExpand<POS> {}
impl<const POS: u8> Recip for FloatExpand<POS> {}
impl<const POS: u8> Erf for FloatExpand<POS> {}
impl<const POS: u8> Exp for FloatExpand<POS> {}
impl<const POS: u8> Remainder for FloatExpand<POS> {}
impl<const POS: u8> Abs for FloatExpand<POS> {}
impl<const POS: u8> Max for FloatExpand<POS> {}
impl<const POS: u8> Min for FloatExpand<POS> {}
impl<const POS: u8> Clamp for FloatExpand<POS> {}
impl<const POS: u8> Log for FloatExpand<POS> {}
impl<const POS: u8> Log1p for FloatExpand<POS> {}
impl<const POS: u8> Cos for FloatExpand<POS> {}
impl<const POS: u8> Sin for FloatExpand<POS> {}
impl<const POS: u8> Tanh for FloatExpand<POS> {}
impl<const POS: u8> Powf for FloatExpand<POS> {}
impl<const POS: u8> Sqrt for FloatExpand<POS> {}
impl<const POS: u8> Round for FloatExpand<POS> {}
impl<const POS: u8> Floor for FloatExpand<POS> {}
impl<const POS: u8> Ceil for FloatExpand<POS> {}

impl<T: Index, const POS: u8> CubeIndex<T> for FloatExpand<POS> {
    type Output = Self;
}
impl<T: Index, const POS: u8> CubeIndexMut<T> for FloatExpand<POS> {}

impl<const POS: u8> Float for FloatExpand<POS> {
    const DIGITS: u32 = 32;

    const EPSILON: Self = FloatExpand::from_f32(half::f16::EPSILON.to_f32_const());

    const INFINITY: Self = FloatExpand::from_f32(f32::INFINITY);

    const MANTISSA_DIGITS: u32 = f32::MANTISSA_DIGITS;

    /// Maximum possible [`tf32`] power of 10 exponent
    const MAX_10_EXP: i32 = f32::MAX_10_EXP;
    /// Maximum possible [`tf32`] power of 2 exponent
    const MAX_EXP: i32 = f32::MAX_EXP;

    /// Minimum possible normal [`tf32`] power of 10 exponent
    const MIN_10_EXP: i32 = f32::MIN_10_EXP;
    /// One greater than the minimum possible normal [`v`] power of 2 exponent
    const MIN_EXP: i32 = f32::MIN_EXP;

    const MIN_POSITIVE: Self = FloatExpand(f32::MIN_POSITIVE);

    const NAN: Self = FloatExpand::from_f32(f32::NAN);

    const NEG_INFINITY: Self = FloatExpand::from_f32(f32::NEG_INFINITY);

    const RADIX: u32 = 2;

    fn new(val: f32) -> Self {
        FloatExpand::from_f32(val)
    }
}

impl<const POS: u8> LaunchArgExpand for FloatExpand<POS> {
    type CompilationArg = ();

    fn expand(_: &Self::CompilationArg, builder: &mut KernelBuilder) -> ExpandElementTyped<Self> {
        builder
            .scalar(FloatExpand::<POS>::as_elem(&builder.context))
            .into()
    }
}

impl<const POS: u8> ScalarArgSettings for FloatExpand<POS> {
    fn register<R: Runtime>(&self, settings: &mut KernelLauncher<R>) {
        settings.register_f32(self.0);
    }
}

impl<const POS: u8> num_traits::Float for FloatExpand<POS> {
    fn nan() -> Self {
        FloatExpand(f32::nan())
    }

    fn infinity() -> Self {
        FloatExpand(f32::infinity())
    }

    fn neg_infinity() -> Self {
        FloatExpand(f32::neg_infinity())
    }

    fn neg_zero() -> Self {
        FloatExpand(f32::neg_zero())
    }

    fn min_value() -> Self {
        FloatExpand(<f32 as num_traits::Float>::min_value())
    }

    fn min_positive_value() -> Self {
        FloatExpand(f32::min_positive_value())
    }

    fn max_value() -> Self {
        FloatExpand(<f32 as num_traits::Float>::max_value())
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
        FloatExpand(self.0.floor())
    }

    fn ceil(self) -> Self {
        FloatExpand(self.0.ceil())
    }

    fn round(self) -> Self {
        FloatExpand(self.0.round())
    }

    fn trunc(self) -> Self {
        FloatExpand(self.0.trunc())
    }

    fn fract(self) -> Self {
        FloatExpand(self.0.fract())
    }

    fn abs(self) -> Self {
        FloatExpand(self.0.abs())
    }

    fn signum(self) -> Self {
        FloatExpand(self.0.signum())
    }

    fn is_sign_positive(self) -> bool {
        self.0.is_sign_positive()
    }

    fn is_sign_negative(self) -> bool {
        self.0.is_sign_negative()
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        FloatExpand(self.0.mul_add(a.0, b.0))
    }

    fn recip(self) -> Self {
        FloatExpand(self.0.recip())
    }

    fn powi(self, n: i32) -> Self {
        FloatExpand(self.0.powi(n))
    }

    fn powf(self, n: Self) -> Self {
        FloatExpand(self.0.powf(n.0))
    }

    fn sqrt(self) -> Self {
        FloatExpand(self.0.sqrt())
    }

    fn exp(self) -> Self {
        FloatExpand(self.0.exp())
    }

    fn exp2(self) -> Self {
        FloatExpand(self.0.exp2())
    }

    fn ln(self) -> Self {
        FloatExpand(self.0.ln())
    }

    fn log(self, base: Self) -> Self {
        FloatExpand(self.0.log(base.0))
    }

    fn log2(self) -> Self {
        FloatExpand(self.0.log2())
    }

    fn log10(self) -> Self {
        FloatExpand(self.0.log10())
    }

    fn max(self, other: Self) -> Self {
        FloatExpand(self.0.max(other.0))
    }

    fn min(self, other: Self) -> Self {
        FloatExpand(self.0.min(other.0))
    }

    fn abs_sub(self, other: Self) -> Self {
        FloatExpand((self.0 - other.0).abs())
    }

    fn cbrt(self) -> Self {
        FloatExpand(self.0.cbrt())
    }

    fn hypot(self, other: Self) -> Self {
        FloatExpand(self.0.hypot(other.0))
    }

    fn sin(self) -> Self {
        FloatExpand(self.0.sin())
    }

    fn cos(self) -> Self {
        FloatExpand(self.0.cos())
    }

    fn tan(self) -> Self {
        FloatExpand(self.0.tan())
    }

    fn asin(self) -> Self {
        FloatExpand(self.0.asin())
    }

    fn acos(self) -> Self {
        FloatExpand(self.0.acos())
    }

    fn atan(self) -> Self {
        FloatExpand(self.0.atan())
    }

    fn atan2(self, other: Self) -> Self {
        FloatExpand(self.0.atan2(other.0))
    }

    fn sin_cos(self) -> (Self, Self) {
        let (a, b) = self.0.sin_cos();
        (FloatExpand(a), FloatExpand(b))
    }

    fn exp_m1(self) -> Self {
        FloatExpand(self.0.exp_m1())
    }

    fn ln_1p(self) -> Self {
        FloatExpand(self.0.ln_1p())
    }

    fn sinh(self) -> Self {
        FloatExpand(self.0.sinh())
    }

    fn cosh(self) -> Self {
        FloatExpand(self.0.cosh())
    }

    fn tanh(self) -> Self {
        FloatExpand(self.0.tanh())
    }

    fn asinh(self) -> Self {
        FloatExpand(self.0.asinh())
    }

    fn acosh(self) -> Self {
        FloatExpand(self.0.acosh())
    }

    fn atanh(self) -> Self {
        FloatExpand(self.0.atanh())
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        self.0.integer_decode()
    }
}

impl<const POS: u8> Num for FloatExpand<POS> {
    type FromStrRadixErr = <f32 as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Ok(FloatExpand(f32::from_str_radix(str, radix)?))
    }
}

impl<const POS: u8> One for FloatExpand<POS> {
    fn one() -> Self {
        FloatExpand(1.0)
    }
}

impl<const POS: u8> Zero for FloatExpand<POS> {
    fn zero() -> Self {
        FloatExpand(0.0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0.0
    }
}
