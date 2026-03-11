use bytemuck::{Pod, Zeroable};
use core::num::ParseIntError;
use core::ops::*;
use cubecl_ir::{ConstantValue, ExpandElement, Scope, Type, Variable};
use derive_more::derive::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Display, Div,
    DivAssign, Mul, MulAssign, Neg, Not, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub,
    SubAssign,
};
use num_traits::{Num, NumCast, One, ToPrimitive, Zero};
use serde::Serialize;

use crate::prelude::*;

use super::Int;

#[repr(transparent)]
#[derive(
    Clone,
    Copy,
    Default,
    Serialize,
    Zeroable,
    Pod,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
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
    Shl,
    ShlAssign,
    Shr,
    ShrAssign,
    BitXor,
    BitXorAssign,
    BitAnd,
    BitAndAssign,
    BitOr,
    BitOrAssign,
    Not,
    Hash,
)]
pub struct IntExpand<const POS: usize>(i64);

impl<const POS: usize> Mul for IntExpand<POS> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        IntExpand(self.0 * rhs.0)
    }
}

impl<const POS: usize> Div for IntExpand<POS> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        IntExpand(self.0 / rhs.0)
    }
}

impl<const POS: usize> Rem for IntExpand<POS> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        IntExpand(self.0 % rhs.0)
    }
}

impl<const POS: usize> MulAssign for IntExpand<POS> {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl<const POS: usize> DivAssign for IntExpand<POS> {
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0;
    }
}

impl<const POS: usize> RemAssign for IntExpand<POS> {
    fn rem_assign(&mut self, rhs: Self) {
        self.0 %= rhs.0;
    }
}

impl<const POS: usize> Shr for IntExpand<POS> {
    type Output = Self;

    fn shr(self, rhs: Self) -> Self::Output {
        IntExpand(self.0 >> rhs.0)
    }
}

impl<const POS: usize> Shl for IntExpand<POS> {
    type Output = Self;

    fn shl(self, rhs: Self) -> Self::Output {
        IntExpand(self.0 << rhs.0)
    }
}

impl<const POS: usize> ToPrimitive for IntExpand<POS> {
    fn to_i64(&self) -> Option<i64> {
        Some(self.0)
    }

    fn to_u64(&self) -> Option<u64> {
        Some(self.0 as u64)
    }

    fn to_f32(&self) -> Option<f32> {
        Some(self.0 as f32)
    }

    fn to_f64(&self) -> Option<f64> {
        Some(self.0 as f64)
    }
}

impl<const POS: usize> NumCast for IntExpand<POS> {
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        Some(IntExpand(n.to_i64()?))
    }
}

impl<const POS: usize> CubeType for IntExpand<POS> {
    type ExpandType = ExpandElementTyped<IntExpand<POS>>;
}

impl<const POS: usize> Scalar for IntExpand<POS> {}
impl<const POS: usize> CubePrimitive for IntExpand<POS> {
    type Scalar = Self;
    type Size = Const<1>;
    type WithScalar<S: Scalar> = S;

    /// Return the element type to use on GPU
    fn as_type(scope: &Scope) -> Type {
        Type::new(scope.resolve_type::<Self>().expect("Type to be registered"))
    }

    fn from_const_value(_value: ConstantValue) -> Self {
        unimplemented!("Can't turn `IntExpand` into a constant value")
    }
}

impl<const POS: usize> From<IntExpand<POS>> for ConstantValue {
    fn from(val: IntExpand<POS>) -> Self {
        val.0.into()
    }
}

impl<const POS: usize> From<IntExpand<POS>> for Variable {
    fn from(val: IntExpand<POS>) -> Self {
        // TODO: Fix how we create literal.
        Variable::constant(val.0.into(), cubecl_ir::IntKind::I64)
    }
}

impl<const POS: usize> From<IntExpand<POS>> for ExpandElementTyped<IntExpand<POS>> {
    fn from(value: IntExpand<POS>) -> Self {
        let var: Variable = value.into();
        ExpandElementTyped::new(ExpandElement::Plain(var))
    }
}

impl<const POS: usize> IntoRuntime for IntExpand<POS> {
    fn __expand_runtime_method(self, scope: &mut Scope) -> ExpandElementTyped<Self> {
        ExpandElementTyped::from_lit(scope, self.0)
    }
}

impl<const POS: usize> Numeric for IntExpand<POS> {
    fn min_value() -> Self {
        panic!("Can't use min value in comptime with dynamic element type");
    }
    fn max_value() -> Self {
        panic!("Can't use max value in comptime with dynamic element type");
    }
}

impl<const POS: usize> ExpandElementAssign for IntExpand<POS> {}

impl<const POS: usize> ScalarArgSettings for IntExpand<POS> {
    fn register<R: Runtime>(&self, _launcher: &mut KernelLauncher<R>) {
        panic!("Can't launch `IntExpand` as scalar")
    }
}

impl<const POS: usize> Remainder for IntExpand<POS> {}
impl<const POS: usize> Abs for IntExpand<POS> {}
impl<const POS: usize> MulHi for IntExpand<POS> {}

impl<const POS: usize> CubeNot for IntExpand<POS> {}
impl<const POS: usize> ReverseBits for IntExpand<POS> {}
impl<const POS: usize> CountOnes for IntExpand<POS> {}
impl<const POS: usize> FindFirstSet for IntExpand<POS> {}
impl<const POS: usize> LeadingZeros for IntExpand<POS> {}
impl<const POS: usize> TrailingZeros for IntExpand<POS> {}
impl<const POS: usize> SaturatingAdd for IntExpand<POS> {}
impl<const POS: usize> SaturatingSub for IntExpand<POS> {}

impl<const POS: usize> Int for IntExpand<POS> {
    const BITS: u32 = 32;

    fn new(val: i64) -> Self {
        IntExpand(val)
    }
}

impl<const POS: usize> Zero for IntExpand<POS> {
    fn zero() -> Self {
        Self(0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl<const POS: usize> One for IntExpand<POS> {
    fn one() -> Self {
        Self(1)
    }
}

impl<const POS: usize> Num for IntExpand<POS> {
    type FromStrRadixErr = ParseIntError;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Ok(IntExpand(i64::from_str_radix(str, radix)?))
    }
}
