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

#![allow(clippy::multiple_bound_locations)]

use core::{
    cmp::Ordering,
    ops::{
        BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign, Mul,
        MulAssign, Neg, Not, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign,
    },
};
use core::{f32, marker::PhantomData};

use bytemuck::{Pod, Zeroable};
use cubecl_ir::ExpandElement;
use derive_more::{
    Deref, DerefMut,
    derive::{
        Add, AddAssign, Debug, Display, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub,
        SubAssign,
    },
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
    Serialize,
    Pod,
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
    Deref,
    DerefMut,
)]

/// A fake element type that can be configured to map to any other element type.
#[display("{val}")]
pub struct ElemExpand<Marker: 'static> {
    #[deref]
    #[deref_mut]
    val: f32,
    // derive_more has no universal attribute unfortunately
    #[add(ignore)]
    #[sub(ignore)]
    #[mul(ignore)]
    #[div(ignore)]
    #[rem(ignore)]
    #[add_assign(ignore)]
    #[sub_assign(ignore)]
    #[mul_assign(ignore)]
    #[div_assign(ignore)]
    #[rem_assign(ignore)]
    #[debug(ignore)]
    _ty: PhantomData<Marker>,
}

unsafe impl<Marker: 'static> Zeroable for ElemExpand<Marker> {}
unsafe impl<Marker: 'static> Send for ElemExpand<Marker> {}
unsafe impl<Marker: 'static> Sync for ElemExpand<Marker> {}

impl<Marker: 'static> Copy for ElemExpand<Marker> {}
impl<Marker: 'static> Clone for ElemExpand<Marker> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Marker: 'static> Default for ElemExpand<Marker> {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl<Marker: 'static> Eq for ElemExpand<Marker> {}
impl<Marker: 'static> PartialEq for ElemExpand<Marker> {
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val
    }
}

/// A fake float element type that can be configured to map to any other element type.
pub type FloatExpand<Marker> = ElemExpand<Marker>;

/// A fake numeric element type that can be configured to map to any other element type.
pub type NumericExpand<Marker> = ElemExpand<Marker>;

/// A fake constant type that can be configured to map to any comptime value.
#[derive(Debug)]
pub struct SizeExpand<Marker> {
    #[debug(ignore)]
    _ty: PhantomData<Marker>,
}

unsafe impl<Marker: 'static> Send for SizeExpand<Marker> {}
unsafe impl<Marker: 'static> Sync for SizeExpand<Marker> {}

impl<Marker: 'static> Copy for SizeExpand<Marker> {}
impl<Marker: 'static> Clone for SizeExpand<Marker> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Marker: 'static> Neg for ElemExpand<Marker> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.val)
    }
}

impl<Marker: 'static> ElemExpand<Marker> {
    pub const MIN_POSITIVE: Self = Self::new(half::f16::MIN_POSITIVE.to_f32_const());

    pub const fn new(val: f32) -> Self {
        Self {
            val,
            _ty: PhantomData,
        }
    }

    pub const fn from_f32(val: f32) -> Self {
        ElemExpand::new(val)
    }

    pub const fn from_f64(val: f64) -> Self {
        ElemExpand::new(val as f32)
    }

    pub const fn to_f32(self) -> f32 {
        self.val
    }

    pub const fn to_f64(self) -> f64 {
        self.val as f64
    }

    pub fn total_cmp(&self, other: &Self) -> Ordering {
        self.val.total_cmp(&other.val)
    }

    pub fn is_nan(&self) -> bool {
        self.val.is_nan()
    }
}

impl<Marker: 'static> Mul for ElemExpand<Marker> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        ElemExpand::new(self.val * rhs.val)
    }
}

impl<Marker: 'static> Div for ElemExpand<Marker> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        ElemExpand::new(self.val / rhs.val)
    }
}

impl<Marker: 'static> Rem for ElemExpand<Marker> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        ElemExpand::new(self.val % rhs.val)
    }
}

impl<Marker: 'static> MulAssign for ElemExpand<Marker> {
    fn mul_assign(&mut self, rhs: Self) {
        self.val *= rhs.val;
    }
}

impl<Marker: 'static> DivAssign for ElemExpand<Marker> {
    fn div_assign(&mut self, rhs: Self) {
        self.val /= rhs.val;
    }
}

impl<Marker: 'static> RemAssign for ElemExpand<Marker> {
    fn rem_assign(&mut self, rhs: Self) {
        self.val %= rhs.val;
    }
}

impl<Marker: 'static> From<f32> for ElemExpand<Marker> {
    fn from(value: f32) -> Self {
        Self::from_f32(value)
    }
}

impl<Marker: 'static> From<ElemExpand<Marker>> for f32 {
    fn from(val: ElemExpand<Marker>) -> Self {
        val.to_f32()
    }
}

impl<Marker: 'static> ToPrimitive for ElemExpand<Marker> {
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

impl<Marker: 'static> NumCast for ElemExpand<Marker> {
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        Some(ElemExpand::from_f32(n.to_f32()?))
    }
}

impl<Marker: 'static> CubeType for ElemExpand<Marker> {
    type ExpandType = ExpandElementTyped<ElemExpand<Marker>>;
}

impl<Marker: 'static> Scalar for ElemExpand<Marker> {}
impl<Marker: 'static> CubePrimitive for ElemExpand<Marker> {
    type Scalar = Self;
    type Size = Const<1>;
    type WithScalar<S: Scalar> = S;

    /// Return the element type to use on GPU
    fn as_type(scope: &Scope) -> Type {
        Type::new(scope.resolve_type::<Self>().expect("Type to be registered"))
    }

    fn from_const_value(_value: ConstantValue) -> Self {
        unimplemented!("Can't turn `ElemExpand` into a constant value")
    }
}

impl<Marker: 'static> From<ElemExpand<Marker>> for ConstantValue {
    fn from(val: ElemExpand<Marker>) -> Self {
        val.val.into()
    }
}

impl<Marker: 'static> From<ElemExpand<Marker>> for Variable {
    fn from(val: ElemExpand<Marker>) -> Self {
        // TODO: Fix how we create literal.
        Variable::constant(val.val.into(), FloatKind::F32)
    }
}

impl<Marker: 'static> From<ElemExpand<Marker>> for ExpandElementTyped<ElemExpand<Marker>> {
    fn from(value: ElemExpand<Marker>) -> Self {
        let var: Variable = value.into();
        ExpandElementTyped::new(ExpandElement::Plain(var))
    }
}

impl<Marker: 'static> IntoRuntime for ElemExpand<Marker> {
    fn __expand_runtime_method(self, scope: &mut Scope) -> ExpandElementTyped<Self> {
        ExpandElementTyped::from_lit(scope, self)
    }
}

impl<Marker: 'static> Numeric for ElemExpand<Marker> {
    fn min_value() -> Self {
        panic!("Can't use min value in comptime with dynamic element type");
    }
    fn max_value() -> Self {
        panic!("Can't use max value in comptime with dynamic element type");
    }
}

impl<Marker: 'static> ExpandElementAssign for ElemExpand<Marker> {}

impl<Marker: 'static> ScalarArgSettings for ElemExpand<Marker> {
    fn register<R: Runtime>(&self, _launcher: &mut KernelLauncher<R>) {
        panic!("Can't launch `ElemExpand` as scalar")
    }
}

impl<Marker: 'static> Normalize for ElemExpand<Marker> {}
impl<Marker: 'static> Dot for ElemExpand<Marker> {}
impl<Marker: 'static> Magnitude for ElemExpand<Marker> {}
impl<Marker: 'static> Recip for ElemExpand<Marker> {}
impl<Marker: 'static> Erf for ElemExpand<Marker> {}
impl<Marker: 'static> Exp for ElemExpand<Marker> {}
impl<Marker: 'static> Remainder for ElemExpand<Marker> {}
impl<Marker: 'static> Abs for ElemExpand<Marker> {}
impl<Marker: 'static> Log for ElemExpand<Marker> {}
impl<Marker: 'static> Log1p for ElemExpand<Marker> {}
impl<Marker: 'static> Cos for ElemExpand<Marker> {}
impl<Marker: 'static> Sin for ElemExpand<Marker> {}
impl<Marker: 'static> Tan for ElemExpand<Marker> {}
impl<Marker: 'static> Tanh for ElemExpand<Marker> {}
impl<Marker: 'static> Sinh for ElemExpand<Marker> {}
impl<Marker: 'static> Cosh for ElemExpand<Marker> {}
impl<Marker: 'static> ArcCos for ElemExpand<Marker> {}
impl<Marker: 'static> ArcSin for ElemExpand<Marker> {}
impl<Marker: 'static> ArcTan for ElemExpand<Marker> {}
impl<Marker: 'static> ArcSinh for ElemExpand<Marker> {}
impl<Marker: 'static> ArcCosh for ElemExpand<Marker> {}
impl<Marker: 'static> ArcTanh for ElemExpand<Marker> {}
impl<Marker: 'static> Degrees for ElemExpand<Marker> {}
impl<Marker: 'static> Radians for ElemExpand<Marker> {}
impl<Marker: 'static> ArcTan2 for ElemExpand<Marker> {}
impl<Marker: 'static> Powf for ElemExpand<Marker> {}
impl<Marker: 'static, I: CubePrimitive> Powi<I> for ElemExpand<Marker> {}
impl<Marker: 'static> Hypot for ElemExpand<Marker> {}
impl<Marker: 'static> Rhypot for ElemExpand<Marker> {}
impl<Marker: 'static> Sqrt for ElemExpand<Marker> {}
impl<Marker: 'static> InverseSqrt for ElemExpand<Marker> {}
impl<Marker: 'static> Round for ElemExpand<Marker> {}
impl<Marker: 'static> Floor for ElemExpand<Marker> {}
impl<Marker: 'static> Ceil for ElemExpand<Marker> {}
impl<Marker: 'static> Trunc for ElemExpand<Marker> {}
impl<Marker: 'static> IsNan for ElemExpand<Marker> {}
impl<Marker: 'static> IsInf for ElemExpand<Marker> {}

#[allow(clippy::non_canonical_partial_ord_impl)]
impl<Marker: 'static> PartialOrd for ElemExpand<Marker> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        FloatOrd(self.val).partial_cmp(&FloatOrd(other.val))
    }
}

impl<Marker: 'static> Ord for ElemExpand<Marker> {
    fn cmp(&self, other: &Self) -> Ordering {
        FloatOrd(self.val).cmp(&FloatOrd(other.val))
    }
}

impl<Marker: 'static> Float for ElemExpand<Marker> {
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

    const MIN_POSITIVE: Self = ElemExpand::new(f32::MIN_POSITIVE);

    const NAN: Self = ElemExpand::from_f32(f32::NAN);

    const NEG_INFINITY: Self = ElemExpand::from_f32(f32::NEG_INFINITY);

    const RADIX: u32 = 2;

    fn new(val: f32) -> Self {
        ElemExpand::from_f32(val)
    }
}

impl<Marker: 'static> Int for ElemExpand<Marker> {
    const BITS: u32 = 32;

    fn new(val: i64) -> Self {
        ElemExpand::from_f32(val as f32)
    }
}
impl<Marker: 'static> CubeNot for ElemExpand<Marker> {}
impl<Marker: 'static> ReverseBits for ElemExpand<Marker> {}
impl<Marker: 'static> CountOnes for ElemExpand<Marker> {}
impl<Marker: 'static> LeadingZeros for ElemExpand<Marker> {}
impl<Marker: 'static> TrailingZeros for ElemExpand<Marker> {}
impl<Marker: 'static> FindFirstSet for ElemExpand<Marker> {}
impl<Marker: 'static> SaturatingAdd for ElemExpand<Marker> {}
impl<Marker: 'static> SaturatingSub for ElemExpand<Marker> {}
impl<Marker: 'static> core::hash::Hash for ElemExpand<Marker> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.val.to_bits().hash(state);
    }
}

impl<Marker: 'static> Not for ElemExpand<Marker> {
    type Output = Self;

    fn not(self) -> Self::Output {
        ElemExpand::new(!(self.val as i64) as f32)
    }
}

impl<Marker: 'static> BitOr for ElemExpand<Marker> {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self::new(BitOr::bitor(self.val as i32, rhs.val as i32) as f32)
    }
}
impl<Marker: 'static> BitXor for ElemExpand<Marker> {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self::new(BitXor::bitxor(self.val as i32, rhs.val as i32) as f32)
    }
}

impl<Marker: 'static> BitAnd for ElemExpand<Marker> {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self::new(BitAnd::bitand(self.val as i32, rhs.val as i32) as f32)
    }
}
impl<Marker: 'static> BitAndAssign for ElemExpand<Marker> {
    fn bitand_assign(&mut self, rhs: Self) {
        let mut value = self.val as i32;
        BitAndAssign::bitand_assign(&mut value, rhs.val as i32);
        self.val = value as f32
    }
}
impl<Marker: 'static> BitXorAssign for ElemExpand<Marker> {
    fn bitxor_assign(&mut self, rhs: Self) {
        let mut value = self.val as i32;
        BitXorAssign::bitxor_assign(&mut value, rhs.val as i32);
        self.val = value as f32
    }
}

impl<Marker: 'static> BitOrAssign for ElemExpand<Marker> {
    fn bitor_assign(&mut self, rhs: Self) {
        let mut value = self.val as i32;
        BitOrAssign::bitor_assign(&mut value, rhs.val as i32);
        self.val = value as f32
    }
}
impl<Marker: 'static> Shl for ElemExpand<Marker> {
    type Output = Self;

    fn shl(self, rhs: Self) -> Self::Output {
        Self::new(Shl::shl(self.val as i32, rhs.val as i32) as f32)
    }
}
impl<Marker: 'static> Shr for ElemExpand<Marker> {
    type Output = Self;

    fn shr(self, rhs: Self) -> Self::Output {
        Self::new(Shr::shr(self.val as i32, rhs.val as i32) as f32)
    }
}

impl<Marker: 'static> ShrAssign<u32> for ElemExpand<Marker> {
    fn shr_assign(&mut self, rhs: u32) {
        let mut value = self.val as i32;
        ShrAssign::shr_assign(&mut value, rhs);
        self.val = value as f32
    }
}

impl<Marker: 'static> ShlAssign<u32> for ElemExpand<Marker> {
    fn shl_assign(&mut self, rhs: u32) {
        let mut value = self.val as i32;
        ShlAssign::shl_assign(&mut value, rhs);
        self.val = value as f32
    }
}

impl<Marker: 'static> num_traits::Float for ElemExpand<Marker> {
    fn nan() -> Self {
        ElemExpand::new(f32::nan())
    }

    fn infinity() -> Self {
        ElemExpand::new(f32::infinity())
    }

    fn neg_infinity() -> Self {
        ElemExpand::new(f32::neg_infinity())
    }

    fn neg_zero() -> Self {
        ElemExpand::new(f32::neg_zero())
    }

    fn min_value() -> Self {
        ElemExpand::new(<f32 as num_traits::Float>::min_value())
    }

    fn min_positive_value() -> Self {
        ElemExpand::new(f32::min_positive_value())
    }

    fn max_value() -> Self {
        ElemExpand::new(<f32 as num_traits::Float>::max_value())
    }

    fn is_nan(self) -> bool {
        self.val.is_nan()
    }

    fn is_infinite(self) -> bool {
        self.val.is_infinite()
    }

    fn is_finite(self) -> bool {
        self.val.is_finite()
    }

    fn is_normal(self) -> bool {
        self.val.is_normal()
    }

    fn classify(self) -> core::num::FpCategory {
        self.val.classify()
    }

    fn floor(self) -> Self {
        ElemExpand::new(self.val.floor())
    }

    fn ceil(self) -> Self {
        ElemExpand::new(self.val.ceil())
    }

    fn round(self) -> Self {
        ElemExpand::new(self.val.round())
    }

    fn trunc(self) -> Self {
        ElemExpand::new(self.val.trunc())
    }

    fn fract(self) -> Self {
        ElemExpand::new(self.val.fract())
    }

    fn abs(self) -> Self {
        ElemExpand::new(self.val.abs())
    }

    fn signum(self) -> Self {
        ElemExpand::new(self.val.signum())
    }

    fn is_sign_positive(self) -> bool {
        self.val.is_sign_positive()
    }

    fn is_sign_negative(self) -> bool {
        self.val.is_sign_negative()
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        ElemExpand::new(self.val.mul_add(a.val, b.val))
    }

    fn recip(self) -> Self {
        ElemExpand::new(self.val.recip())
    }

    fn powi(self, n: i32) -> Self {
        ElemExpand::new(self.val.powi(n))
    }

    fn powf(self, n: Self) -> Self {
        ElemExpand::new(self.val.powf(n.val))
    }

    fn sqrt(self) -> Self {
        ElemExpand::new(self.val.sqrt())
    }

    fn exp(self) -> Self {
        ElemExpand::new(self.val.exp())
    }

    fn exp2(self) -> Self {
        ElemExpand::new(self.val.exp2())
    }

    fn ln(self) -> Self {
        ElemExpand::new(self.val.ln())
    }

    fn log(self, base: Self) -> Self {
        ElemExpand::new(self.val.log(base.val))
    }

    fn log2(self) -> Self {
        ElemExpand::new(self.val.log2())
    }

    fn log10(self) -> Self {
        ElemExpand::new(self.val.log10())
    }

    fn max(self, other: Self) -> Self {
        ElemExpand::new(self.val.max(other.val))
    }

    fn min(self, other: Self) -> Self {
        ElemExpand::new(self.val.min(other.val))
    }

    fn abs_sub(self, other: Self) -> Self {
        ElemExpand::new((self.val - other.val).abs())
    }

    fn cbrt(self) -> Self {
        ElemExpand::new(self.val.cbrt())
    }

    fn hypot(self, other: Self) -> Self {
        ElemExpand::new(self.val.hypot(other.val))
    }

    fn sin(self) -> Self {
        ElemExpand::new(self.val.sin())
    }

    fn cos(self) -> Self {
        ElemExpand::new(self.val.cos())
    }

    fn tan(self) -> Self {
        ElemExpand::new(self.val.tan())
    }

    fn asin(self) -> Self {
        ElemExpand::new(self.val.asin())
    }

    fn acos(self) -> Self {
        ElemExpand::new(self.val.acos())
    }

    fn atan(self) -> Self {
        ElemExpand::new(self.val.atan())
    }

    fn atan2(self, other: Self) -> Self {
        ElemExpand::new(self.val.atan2(other.val))
    }

    fn sin_cos(self) -> (Self, Self) {
        let (a, b) = self.val.sin_cos();
        (ElemExpand::new(a), ElemExpand::new(b))
    }

    fn exp_m1(self) -> Self {
        ElemExpand::new(self.val.exp_m1())
    }

    fn ln_1p(self) -> Self {
        ElemExpand::new(self.val.ln_1p())
    }

    fn sinh(self) -> Self {
        ElemExpand::new(self.val.sinh())
    }

    fn cosh(self) -> Self {
        ElemExpand::new(self.val.cosh())
    }

    fn tanh(self) -> Self {
        ElemExpand::new(self.val.tanh())
    }

    fn asinh(self) -> Self {
        ElemExpand::new(self.val.asinh())
    }

    fn acosh(self) -> Self {
        ElemExpand::new(self.val.acosh())
    }

    fn atanh(self) -> Self {
        ElemExpand::new(self.val.atanh())
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        self.val.integer_decode()
    }
}

impl<Marker: 'static> Num for ElemExpand<Marker> {
    type FromStrRadixErr = <f32 as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Ok(ElemExpand::new(f32::from_str_radix(str, radix)?))
    }
}

impl<Marker: 'static> One for ElemExpand<Marker> {
    fn one() -> Self {
        ElemExpand::new(f32::one())
    }
}

impl<Marker: 'static> Zero for ElemExpand<Marker> {
    fn zero() -> Self {
        ElemExpand::new(f32::zero())
    }

    fn is_zero(&self) -> bool {
        self.val.is_zero()
    }
}
