//! This module contains a configurable [element type](DynamicScalar) for floats to be used during
//! kernel expansion to speed up Rust compilation.
//!
//! Expand functions don't need to be generated for different element types even if they are generic
//! over one, since the only use of numeric element types is to map to the [elem IR enum](Elem).
//!
//! This can be done dynamically using the scope instead, reducing the binary size and the
//! compilation time of kernels significantly.
//!
//! You can still have multiple element types in a single kernel, since [`DynamicScalar`] uses const
//! generics to differentiate between float kinds.

#![allow(clippy::multiple_bound_locations)]

use core::{cmp::Ordering, ops::*};
use core::{f32, marker::PhantomData};

use bytemuck::Zeroable;
use cubecl_ir::{ConstantValue, ManagedVariable, Type};
use derive_more::derive::{Debug, Display};
use float_ord::FloatOrd;
use num_traits::{Num, NumCast, One, ToPrimitive, Zero};
use serde::Serialize;

use crate::{
    ir::{FloatKind, Scope, Variable},
    prelude::*,
};

use super::*;

#[allow(non_camel_case_types)]
#[repr(transparent)]
#[derive(Serialize, Debug, Display)]

/// A fake element type that can be configured to map to any other element type.
#[display("{val}")]
pub struct DynamicScalar<Marker: 'static> {
    val: ConstantValue,
    #[debug(ignore)]
    _ty: PhantomData<Marker>,
}

unsafe impl<Marker: 'static> Zeroable for DynamicScalar<Marker> {}
unsafe impl<Marker: 'static> Send for DynamicScalar<Marker> {}
unsafe impl<Marker: 'static> Sync for DynamicScalar<Marker> {}

impl<Marker: 'static> Copy for DynamicScalar<Marker> {}
impl<Marker: 'static> Clone for DynamicScalar<Marker> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Marker: 'static> Default for DynamicScalar<Marker> {
    fn default() -> Self {
        Self::new(ConstantValue::Float(0.0))
    }
}

impl<Marker: 'static> Eq for DynamicScalar<Marker> {}
impl<Marker: 'static> PartialEq for DynamicScalar<Marker> {
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val
    }
}

/// A fake constant type that can be configured to map to any comptime value.
#[derive(Debug)]
pub struct DynamicSize<Marker> {
    #[debug(ignore)]
    _ty: PhantomData<Marker>,
}

unsafe impl<Marker: 'static> Send for DynamicSize<Marker> {}
unsafe impl<Marker: 'static> Sync for DynamicSize<Marker> {}

impl<Marker: 'static> Copy for DynamicSize<Marker> {}
impl<Marker: 'static> Clone for DynamicSize<Marker> {
    fn clone(&self) -> Self {
        *self
    }
}

macro_rules! numeric_binop {
    ($op: path, $this: expr, $other: expr) => {
        Self::new(match $this.val {
            ConstantValue::Int(this) => $op(this, $other.val.as_i64()).into(),
            ConstantValue::Float(this) => $op(this, $other.val.as_f64()).into(),
            ConstantValue::UInt(this) => $op(this, $other.val.as_u64()).into(),
            _ => panic!("Mismatched scalar types"),
        })
    };
}

macro_rules! bitwise_binop {
    ($op: path, $this: expr, $other: expr) => {
        Self::new(match $this.val {
            ConstantValue::Int(this) => $op(this, $other.val.as_i64()).into(),
            ConstantValue::Float(this) => {
                (f64::from_bits($op(this.to_bits(), $other.val.as_f64().to_bits()))).into()
            }
            ConstantValue::UInt(this) => $op(this, $other.val.as_u64()).into(),
            _ => panic!("Mismatched scalar types"),
        })
    };
}

impl<Marker: 'static> Neg for DynamicScalar<Marker> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(match self.val {
            ConstantValue::Int(val) => (-val).into(),
            ConstantValue::Float(val) => (-val).into(),
            ConstantValue::UInt(val) => (-(val as i64)).into(),
            ConstantValue::Bool(val) => (!val).into(),
            ConstantValue::Complex(_, _) => panic!("Complex values aren't supported"),
        })
    }
}

impl<Marker: 'static> DynamicScalar<Marker> {
    pub const fn new(val: ConstantValue) -> Self {
        Self {
            val,
            _ty: PhantomData,
        }
    }

    pub const fn from_f64(val: f64) -> Self {
        Self::new(ConstantValue::Float(val))
    }
}

impl<Marker: 'static> Add for DynamicScalar<Marker> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        numeric_binop!(Add::add, self, rhs)
    }
}

impl<Marker: 'static> Sub for DynamicScalar<Marker> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        numeric_binop!(Sub::sub, self, rhs)
    }
}

impl<Marker: 'static> Mul for DynamicScalar<Marker> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        numeric_binop!(Mul::mul, self, rhs)
    }
}

impl<Marker: 'static> Div for DynamicScalar<Marker> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        numeric_binop!(Div::div, self, rhs)
    }
}

impl<Marker: 'static> Rem for DynamicScalar<Marker> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        numeric_binop!(Rem::rem, self, rhs)
    }
}

impl<Marker: 'static> AddAssign for DynamicScalar<Marker> {
    fn add_assign(&mut self, rhs: Self) {
        *self = numeric_binop!(Add::add, *self, rhs);
    }
}

impl<Marker: 'static> SubAssign for DynamicScalar<Marker> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = numeric_binop!(Sub::sub, *self, rhs);
    }
}

impl<Marker: 'static> MulAssign for DynamicScalar<Marker> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = numeric_binop!(Mul::mul, *self, rhs);
    }
}

impl<Marker: 'static> DivAssign for DynamicScalar<Marker> {
    fn div_assign(&mut self, rhs: Self) {
        *self = numeric_binop!(Div::div, *self, rhs);
    }
}

impl<Marker: 'static> RemAssign for DynamicScalar<Marker> {
    fn rem_assign(&mut self, rhs: Self) {
        *self = numeric_binop!(Rem::rem, *self, rhs);
    }
}

impl<Marker: 'static> ToPrimitive for DynamicScalar<Marker> {
    fn to_i64(&self) -> Option<i64> {
        self.val.try_as_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.val.try_as_u64()
    }

    fn to_f32(&self) -> Option<f32> {
        self.val.try_as_f64().map(|it| it as f32)
    }

    fn to_f64(&self) -> Option<f64> {
        self.val.try_as_f64()
    }
}

impl<Marker: 'static> NumCast for DynamicScalar<Marker> {
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        let val = if let Some(float) = n.to_f64() {
            ConstantValue::Float(float)
        } else if let Some(int) = n.to_i64() {
            ConstantValue::Int(int)
        } else if let Some(uint) = n.to_u64() {
            ConstantValue::UInt(uint)
        } else {
            panic!("Unrepresentable value")
        };
        Some(DynamicScalar::new(val))
    }
}

impl<Marker: 'static> CubeType for DynamicScalar<Marker> {
    type ExpandType = NativeExpand<DynamicScalar<Marker>>;
}

impl<Marker: 'static> Scalar for DynamicScalar<Marker> {}
impl<Marker: 'static> CubePrimitive for DynamicScalar<Marker> {
    type Scalar = Self;
    type Size = Const<1>;
    type WithScalar<S: Scalar> = S;

    /// Return the element type to use on GPU
    fn as_type(scope: &Scope) -> Type {
        Type::new(scope.resolve_type::<Self>().expect("Type to be registered"))
    }

    fn from_const_value(value: ConstantValue) -> Self {
        Self::new(value)
    }
}

impl<Marker: 'static> From<DynamicScalar<Marker>> for ConstantValue {
    fn from(val: DynamicScalar<Marker>) -> Self {
        val.val
    }
}

impl<Marker: 'static> From<DynamicScalar<Marker>> for Variable {
    fn from(val: DynamicScalar<Marker>) -> Self {
        // TODO: Fix how we create literal.
        Variable::constant(val.val, FloatKind::F32)
    }
}

impl<Marker: 'static> From<DynamicScalar<Marker>> for NativeExpand<DynamicScalar<Marker>> {
    fn from(value: DynamicScalar<Marker>) -> Self {
        let var: Variable = value.into();
        NativeExpand::new(ManagedVariable::Plain(var))
    }
}

impl<Marker: 'static> IntoRuntime for DynamicScalar<Marker> {
    fn __expand_runtime_method(self, scope: &mut Scope) -> NativeExpand<Self> {
        NativeExpand::from_lit(scope, self)
    }
}

impl<Marker: 'static> Numeric for DynamicScalar<Marker> {
    fn min_value() -> Self {
        panic!("Can't use min value in comptime with dynamic element type");
    }
    fn max_value() -> Self {
        panic!("Can't use max value in comptime with dynamic element type");
    }
}

impl<Marker: 'static> NativeAssign for DynamicScalar<Marker> {}

impl<Marker: 'static> ScalarArgSettings for DynamicScalar<Marker> {
    fn register<R: Runtime>(&self, _launcher: &mut KernelLauncher<R>) {
        panic!("Can't launch `DynamicScalar` as scalar")
    }
}

impl<Marker: 'static> Normalize for DynamicScalar<Marker> {}
impl<Marker: 'static> Dot for DynamicScalar<Marker> {}
impl<Marker: 'static> Magnitude for DynamicScalar<Marker> {}
impl<Marker: 'static> VectorSum for DynamicScalar<Marker> {}
impl<Marker: 'static> Recip for DynamicScalar<Marker> {}
impl<Marker: 'static> Erf for DynamicScalar<Marker> {}
impl<Marker: 'static> Exp for DynamicScalar<Marker> {}
impl<Marker: 'static> Remainder for DynamicScalar<Marker> {}
impl<Marker: 'static> Abs for DynamicScalar<Marker> {
    type AbsElem = Self;
}
impl<Marker: 'static> Log for DynamicScalar<Marker> {}
impl<Marker: 'static> Log1p for DynamicScalar<Marker> {}
impl<Marker: 'static> Expm1 for DynamicScalar<Marker> {}
impl<Marker: 'static> Cos for DynamicScalar<Marker> {}
impl<Marker: 'static> Sin for DynamicScalar<Marker> {}
impl<Marker: 'static> Tan for DynamicScalar<Marker> {}
impl<Marker: 'static> Tanh for DynamicScalar<Marker> {}
impl<Marker: 'static> Sinh for DynamicScalar<Marker> {}
impl<Marker: 'static> Cosh for DynamicScalar<Marker> {}
impl<Marker: 'static> ArcCos for DynamicScalar<Marker> {}
impl<Marker: 'static> ArcSin for DynamicScalar<Marker> {}
impl<Marker: 'static> ArcTan for DynamicScalar<Marker> {}
impl<Marker: 'static> ArcSinh for DynamicScalar<Marker> {}
impl<Marker: 'static> ArcCosh for DynamicScalar<Marker> {}
impl<Marker: 'static> ArcTanh for DynamicScalar<Marker> {}
impl<Marker: 'static> Degrees for DynamicScalar<Marker> {}
impl<Marker: 'static> Radians for DynamicScalar<Marker> {}
impl<Marker: 'static> ArcTan2 for DynamicScalar<Marker> {}
impl<Marker: 'static> Powf for DynamicScalar<Marker> {}
impl<Marker: 'static, I: CubePrimitive> Powi<I> for DynamicScalar<Marker> {}
impl<Marker: 'static> Hypot for DynamicScalar<Marker> {}
impl<Marker: 'static> Rhypot for DynamicScalar<Marker> {}
impl<Marker: 'static> Sqrt for DynamicScalar<Marker> {}
impl<Marker: 'static> InverseSqrt for DynamicScalar<Marker> {}
impl<Marker: 'static> Round for DynamicScalar<Marker> {}
impl<Marker: 'static> Floor for DynamicScalar<Marker> {}
impl<Marker: 'static> Ceil for DynamicScalar<Marker> {}
impl<Marker: 'static> Trunc for DynamicScalar<Marker> {}
impl<Marker: 'static> IsNan for DynamicScalar<Marker> {}
impl<Marker: 'static> IsInf for DynamicScalar<Marker> {}

impl<Marker: 'static> PartialOrd for DynamicScalar<Marker> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<Marker: 'static> Ord for DynamicScalar<Marker> {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.val, other.val) {
            (ConstantValue::Int(this), ConstantValue::Int(other)) => this.cmp(&other),
            (ConstantValue::Float(this), ConstantValue::Float(other)) => {
                FloatOrd(this).cmp(&FloatOrd(other))
            }
            (ConstantValue::UInt(this), ConstantValue::UInt(other)) => this.cmp(&other),
            (ConstantValue::Bool(this), ConstantValue::Bool(other)) => this.cmp(&other),
            _ => panic!("value type mismatch"),
        }
    }
}

impl<Marker: 'static> Int for DynamicScalar<Marker> {
    const BITS: u32 = 32;

    fn new(val: i64) -> Self {
        Self::new(val.into())
    }
}

impl<Marker: 'static> Float for DynamicScalar<Marker> {
    const DIGITS: u32 = 32;
    const EPSILON: Self = DynamicScalar::from_f64(half::f16::EPSILON.to_f64_const());

    const INFINITY: Self = DynamicScalar::from_f64(f64::INFINITY);

    const MANTISSA_DIGITS: u32 = f32::MANTISSA_DIGITS;

    /// Maximum possible [`cubecl_common::tf32`] power of 10 exponent
    const MAX_10_EXP: i32 = f32::MAX_10_EXP;
    /// Maximum possible [`cubecl_common::tf32`] power of 2 exponent
    const MAX_EXP: i32 = f32::MAX_EXP;

    /// Minimum possible normal [`cubecl_common::tf32`] power of 10 exponent
    const MIN_10_EXP: i32 = f32::MIN_10_EXP;

    /// One greater than the minimum possible normal [`cubecl_common::tf32`] power of 2 exponent
    const MIN_EXP: i32 = f32::MIN_EXP;

    const MIN_POSITIVE: Self = DynamicScalar::from_f64(half::bf16::MIN_POSITIVE.to_f64_const());

    const NAN: Self = DynamicScalar::from_f64(f64::NAN);

    const NEG_INFINITY: Self = DynamicScalar::from_f64(f64::NEG_INFINITY);

    const RADIX: u32 = 2;

    fn new(val: f32) -> Self {
        DynamicScalar::from_f64(val as f64)
    }
}

impl<Marker: 'static> CubeNot for DynamicScalar<Marker> {}
impl<Marker: 'static> ReverseBits for DynamicScalar<Marker> {}
impl<Marker: 'static> CountOnes for DynamicScalar<Marker> {}
impl<Marker: 'static> LeadingZeros for DynamicScalar<Marker> {}
impl<Marker: 'static> TrailingZeros for DynamicScalar<Marker> {}
impl<Marker: 'static> FindFirstSet for DynamicScalar<Marker> {}
impl<Marker: 'static> SaturatingAdd for DynamicScalar<Marker> {}
impl<Marker: 'static> SaturatingSub for DynamicScalar<Marker> {}
impl<Marker: 'static> core::hash::Hash for DynamicScalar<Marker> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.val.hash(state);
    }
}

impl<Marker: 'static> Not for DynamicScalar<Marker> {
    type Output = Self;

    fn not(self) -> Self::Output {
        DynamicScalar::new(match self.val {
            ConstantValue::Int(val) => (!val).into(),
            ConstantValue::UInt(val) => (!val).into(),
            ConstantValue::Bool(val) => (!val).into(),
            ConstantValue::Float(val) => f64::from_bits(!val.to_bits()).into(),
            ConstantValue::Complex(_, _) => panic!("Complex values aren't supported"),
        })
    }
}

impl<Marker: 'static> BitOr for DynamicScalar<Marker> {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        bitwise_binop!(BitOr::bitor, self, rhs)
    }
}
impl<Marker: 'static> BitXor for DynamicScalar<Marker> {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        bitwise_binop!(BitXor::bitxor, self, rhs)
    }
}

impl<Marker: 'static> BitAnd for DynamicScalar<Marker> {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        bitwise_binop!(BitAnd::bitand, self, rhs)
    }
}
impl<Marker: 'static> BitAndAssign for DynamicScalar<Marker> {
    fn bitand_assign(&mut self, rhs: Self) {
        *self = bitwise_binop!(BitAnd::bitand, *self, rhs);
    }
}
impl<Marker: 'static> BitXorAssign for DynamicScalar<Marker> {
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = bitwise_binop!(BitXor::bitxor, *self, rhs);
    }
}

impl<Marker: 'static> BitOrAssign for DynamicScalar<Marker> {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = bitwise_binop!(BitOr::bitor, *self, rhs);
    }
}
impl<Marker: 'static> Shl for DynamicScalar<Marker> {
    type Output = Self;

    fn shl(self, rhs: Self) -> Self::Output {
        Self::new(match self.val {
            ConstantValue::Int(val) => (val << rhs.val.as_u64()).into(),
            ConstantValue::Float(val) => f64::from_bits(val.to_bits() << rhs.val.as_u64()).into(),
            ConstantValue::UInt(val) => (val << rhs.val.as_u64()).into(),
            ConstantValue::Bool(_) => panic!("Invalid value"),
            ConstantValue::Complex(_, _) => panic!("Complex values aren't supported"),
        })
    }
}
impl<Marker: 'static> Shr for DynamicScalar<Marker> {
    type Output = Self;

    fn shr(self, rhs: Self) -> Self::Output {
        Self::new(match self.val {
            ConstantValue::Int(val) => (val >> rhs.val.as_u64()).into(),
            ConstantValue::Float(val) => f64::from_bits(val.to_bits() >> rhs.val.as_u64()).into(),
            ConstantValue::UInt(val) => (val >> rhs.val.as_u64()).into(),
            ConstantValue::Bool(_) => panic!("Invalid value"),
            ConstantValue::Complex(_, _) => panic!("Complex values aren't supported"),
        })
    }
}

impl<Marker: 'static> ShrAssign<u32> for DynamicScalar<Marker> {
    fn shr_assign(&mut self, rhs: u32) {
        *self = Self::new(match self.val {
            ConstantValue::Int(val) => (val >> rhs).into(),
            ConstantValue::Float(val) => f64::from_bits(val.to_bits() >> rhs).into(),
            ConstantValue::UInt(val) => (val >> rhs).into(),
            ConstantValue::Bool(_) => panic!("Invalid value"),
            ConstantValue::Complex(_, _) => panic!("Complex values aren't supported"),
        });
    }
}

impl<Marker: 'static> ShlAssign<u32> for DynamicScalar<Marker> {
    fn shl_assign(&mut self, rhs: u32) {
        *self = Self::new(match self.val {
            ConstantValue::Int(val) => (val << rhs).into(),
            ConstantValue::Float(val) => f64::from_bits(val.to_bits() << rhs).into(),
            ConstantValue::UInt(val) => (val << rhs).into(),
            ConstantValue::Bool(_) => panic!("Invalid value"),
            ConstantValue::Complex(_, _) => panic!("Complex values aren't supported"),
        });
    }
}

impl<Marker: 'static> num_traits::Float for DynamicScalar<Marker> {
    fn nan() -> Self {
        DynamicScalar::from_f64(f64::nan())
    }

    fn infinity() -> Self {
        DynamicScalar::from_f64(f64::infinity())
    }

    fn neg_infinity() -> Self {
        DynamicScalar::from_f64(f64::neg_infinity())
    }

    fn neg_zero() -> Self {
        DynamicScalar::from_f64(f64::neg_zero())
    }

    fn min_value() -> Self {
        DynamicScalar::from_f64(<f64 as num_traits::Float>::min_value())
    }

    fn min_positive_value() -> Self {
        DynamicScalar::from_f64(f64::min_positive_value())
    }

    fn max_value() -> Self {
        DynamicScalar::from_f64(<f64 as num_traits::Float>::max_value())
    }

    fn is_nan(self) -> bool {
        match self.val {
            ConstantValue::Float(val) => val.is_nan(),
            _ => false,
        }
    }

    fn is_infinite(self) -> bool {
        match self.val {
            ConstantValue::Float(val) => val.is_infinite(),
            _ => false,
        }
    }

    fn is_finite(self) -> bool {
        match self.val {
            ConstantValue::Float(val) => val.is_finite(),
            _ => true,
        }
    }

    fn is_normal(self) -> bool {
        match self.val {
            ConstantValue::Float(val) => val.is_normal(),
            _ => true,
        }
    }

    fn classify(self) -> core::num::FpCategory {
        match self.val {
            ConstantValue::Float(val) => val.classify(),
            _ => core::num::FpCategory::Normal,
        }
    }

    fn floor(self) -> Self {
        match self.val {
            ConstantValue::Float(val) => Self::from_f64(val.floor()),
            _ => self,
        }
    }

    fn ceil(self) -> Self {
        match self.val {
            ConstantValue::Float(val) => Self::from_f64(val.ceil()),
            _ => self,
        }
    }

    fn round(self) -> Self {
        match self.val {
            ConstantValue::Float(val) => Self::from_f64(val.round()),
            _ => self,
        }
    }

    fn trunc(self) -> Self {
        match self.val {
            ConstantValue::Float(val) => Self::from_f64(val.trunc()),
            _ => self,
        }
    }

    fn fract(self) -> Self {
        match self.val {
            ConstantValue::Float(val) => Self::from_f64(val.fract()),
            _ => self,
        }
    }

    fn abs(self) -> Self {
        match self.val {
            ConstantValue::Float(val) => Self::from_f64(val.abs()),
            _ => self,
        }
    }

    fn signum(self) -> Self {
        match self.val {
            ConstantValue::Float(val) => Self::from_f64(val.signum()),
            _ => self,
        }
    }

    fn is_sign_positive(self) -> bool {
        match self.val {
            ConstantValue::Float(val) => val.is_sign_positive(),
            ConstantValue::Int(val) => val.is_positive(),
            _ => true,
        }
    }

    fn is_sign_negative(self) -> bool {
        match self.val {
            ConstantValue::Float(val) => val.is_sign_negative(),
            ConstantValue::Int(val) => val.is_negative(),
            _ => false,
        }
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
    }

    fn recip(self) -> Self {
        match self.val {
            ConstantValue::Float(val) => Self::from_f64(val.recip()),
            _ => self,
        }
    }

    fn powi(self, n: i32) -> Self {
        match self.val {
            ConstantValue::Float(val) => Self::from_f64(val.powi(n)),
            _ => self,
        }
    }

    fn powf(self, n: Self) -> Self {
        match self.val {
            ConstantValue::Float(val) => Self::from_f64(val.powf(n.val.as_f64())),
            _ => self,
        }
    }

    fn sqrt(self) -> Self {
        match self.val {
            ConstantValue::Float(val) => Self::from_f64(val.sqrt()),
            _ => self,
        }
    }

    fn exp(self) -> Self {
        match self.val {
            ConstantValue::Float(val) => Self::from_f64(val.exp()),
            _ => self,
        }
    }

    fn exp2(self) -> Self {
        match self.val {
            ConstantValue::Float(val) => Self::from_f64(val.exp2()),
            _ => self,
        }
    }

    fn ln(self) -> Self {
        match self.val {
            ConstantValue::Float(val) => Self::from_f64(val.ln()),
            _ => self,
        }
    }

    fn log(self, base: Self) -> Self {
        match self.val {
            ConstantValue::Float(val) => Self::from_f64(val.log(base.val.as_f64())),
            _ => self,
        }
    }

    fn log2(self) -> Self {
        match self.val {
            ConstantValue::Float(val) => Self::from_f64(val.log2()),
            _ => self,
        }
    }

    fn log10(self) -> Self {
        match self.val {
            ConstantValue::Float(val) => Self::from_f64(val.log10()),
            _ => self,
        }
    }

    fn max(self, other: Self) -> Self {
        DynamicScalar::new(self.val.max(other.val))
    }

    fn min(self, other: Self) -> Self {
        DynamicScalar::new(self.val.min(other.val))
    }

    fn abs_sub(self, other: Self) -> Self {
        num_traits::Float::abs(self - other)
    }

    fn cbrt(self) -> Self {
        DynamicScalar::from_f64(self.val.as_f64().cbrt())
    }

    fn hypot(self, other: Self) -> Self {
        DynamicScalar::from_f64(self.val.as_f64().hypot(other.val.as_f64()))
    }

    fn sin(self) -> Self {
        DynamicScalar::from_f64(self.val.as_f64().sin())
    }

    fn cos(self) -> Self {
        DynamicScalar::from_f64(self.val.as_f64().cos())
    }

    fn tan(self) -> Self {
        DynamicScalar::from_f64(self.val.as_f64().tan())
    }

    fn asin(self) -> Self {
        DynamicScalar::from_f64(self.val.as_f64().asin())
    }

    fn acos(self) -> Self {
        DynamicScalar::from_f64(self.val.as_f64().acos())
    }

    fn atan(self) -> Self {
        DynamicScalar::from_f64(self.val.as_f64().atan())
    }

    fn atan2(self, other: Self) -> Self {
        DynamicScalar::from_f64(self.val.as_f64().atan2(other.val.as_f64()))
    }

    fn sin_cos(self) -> (Self, Self) {
        let (a, b) = self.val.as_f64().sin_cos();
        (DynamicScalar::from_f64(a), DynamicScalar::from_f64(b))
    }

    fn exp_m1(self) -> Self {
        DynamicScalar::from_f64(self.val.as_f64().exp_m1())
    }

    fn ln_1p(self) -> Self {
        DynamicScalar::from_f64(self.val.as_f64().ln_1p())
    }

    fn sinh(self) -> Self {
        DynamicScalar::from_f64(self.val.as_f64().sinh())
    }

    fn cosh(self) -> Self {
        DynamicScalar::from_f64(self.val.as_f64().cosh())
    }

    fn tanh(self) -> Self {
        DynamicScalar::from_f64(self.val.as_f64().tanh())
    }

    fn asinh(self) -> Self {
        DynamicScalar::from_f64(self.val.as_f64().asinh())
    }

    fn acosh(self) -> Self {
        DynamicScalar::from_f64(self.val.as_f64().acosh())
    }

    fn atanh(self) -> Self {
        DynamicScalar::from_f64(self.val.as_f64().atanh())
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        self.val.as_f64().integer_decode()
    }
}

impl<Marker: 'static> Num for DynamicScalar<Marker> {
    type FromStrRadixErr = <f32 as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Ok(DynamicScalar::from_f64(f64::from_str_radix(str, radix)?))
    }
}

impl<Marker: 'static> One for DynamicScalar<Marker> {
    fn one() -> Self {
        DynamicScalar::new(u64::one().into())
    }
}

impl<Marker: 'static> Zero for DynamicScalar<Marker> {
    fn zero() -> Self {
        DynamicScalar::new(u64::zero().into())
    }

    fn is_zero(&self) -> bool {
        self.val.is_zero()
    }
}

impl<Marker: 'static> ComplexCore for DynamicScalar<Marker> {
    type FloatElem = Self;
}

impl<Marker: 'static> ComplexCompare for DynamicScalar<Marker> {}

impl<Marker: 'static> ComplexMath for DynamicScalar<Marker> {}
