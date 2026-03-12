#![allow(clippy::multiple_bound_locations)]

use bytemuck::{Pod, Zeroable};
use core::ops::*;
use core::{marker::PhantomData, num::ParseIntError};
use cubecl_ir::{ConstantValue, ExpandElement, Scope, Type, Variable};
use derive_more::{
    Deref, DerefMut,
    derive::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Debug,
        Display, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign,
        Sub, SubAssign,
    },
};
use num_traits::{Num, NumCast, One, ToPrimitive, Zero};
use serde::Serialize;

use crate::prelude::*;

use super::Int;

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
    Deref,
    DerefMut,
)]
#[display("{val}")]
pub struct IntExpand<Marker: 'static> {
    #[deref]
    #[deref_mut]
    val: i64,
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
    #[shl(ignore)]
    #[shl_assign(ignore)]
    #[shr(ignore)]
    #[shr_assign(ignore)]
    #[bitxor(ignore)]
    #[bitxor_assign(ignore)]
    #[bitand(ignore)]
    #[bitand_assign(ignore)]
    #[bitor(ignore)]
    #[bitor_assign(ignore)]
    _ty: PhantomData<Marker>,
}

unsafe impl<Marker: 'static> Zeroable for IntExpand<Marker> {}
unsafe impl<Marker: 'static> Send for IntExpand<Marker> {}
unsafe impl<Marker: 'static> Sync for IntExpand<Marker> {}

impl<Marker: 'static> Copy for IntExpand<Marker> {}
impl<Marker: 'static> Clone for IntExpand<Marker> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Marker: 'static> Default for IntExpand<Marker> {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl<Marker: 'static> Eq for IntExpand<Marker> {}
impl<Marker: 'static> PartialEq for IntExpand<Marker> {
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val
    }
}

impl<Marker: 'static> Ord for IntExpand<Marker> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.val.cmp(&other.val)
    }
}
impl<Marker: 'static> PartialOrd for IntExpand<Marker> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<Marker: 'static> core::hash::Hash for IntExpand<Marker> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.val.hash(state);
    }
}

impl<Marker: 'static> IntExpand<Marker> {
    pub const fn new(val: i64) -> Self {
        Self {
            val,
            _ty: PhantomData,
        }
    }
}

impl<Marker: 'static> Not for IntExpand<Marker> {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self::new(!self.val)
    }
}

impl<Marker: 'static> Neg for IntExpand<Marker> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.val)
    }
}

impl<Marker: 'static> Mul for IntExpand<Marker> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        IntExpand::new(self.val * rhs.val)
    }
}

impl<Marker: 'static> Div for IntExpand<Marker> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        IntExpand::new(self.val / rhs.val)
    }
}

impl<Marker: 'static> Rem for IntExpand<Marker> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        IntExpand::new(self.val % rhs.val)
    }
}

impl<Marker: 'static> MulAssign for IntExpand<Marker> {
    fn mul_assign(&mut self, rhs: Self) {
        self.val *= rhs.val;
    }
}

impl<Marker: 'static> DivAssign for IntExpand<Marker> {
    fn div_assign(&mut self, rhs: Self) {
        self.val /= rhs.val;
    }
}

impl<Marker: 'static> RemAssign for IntExpand<Marker> {
    fn rem_assign(&mut self, rhs: Self) {
        self.val %= rhs.val;
    }
}

impl<Marker: 'static> Shr for IntExpand<Marker> {
    type Output = Self;

    fn shr(self, rhs: Self) -> Self::Output {
        IntExpand::new(self.val >> rhs.val)
    }
}

impl<Marker: 'static> Shl for IntExpand<Marker> {
    type Output = Self;

    fn shl(self, rhs: Self) -> Self::Output {
        IntExpand::new(self.val << rhs.val)
    }
}

impl<Marker: 'static> ToPrimitive for IntExpand<Marker> {
    fn to_i64(&self) -> Option<i64> {
        Some(self.val)
    }

    fn to_u64(&self) -> Option<u64> {
        Some(self.val as u64)
    }

    fn to_f32(&self) -> Option<f32> {
        Some(self.val as f32)
    }

    fn to_f64(&self) -> Option<f64> {
        Some(self.val as f64)
    }
}

impl<Marker: 'static> NumCast for IntExpand<Marker> {
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        Some(IntExpand::new(n.to_i64()?))
    }
}

impl<Marker: 'static> CubeType for IntExpand<Marker> {
    type ExpandType = ExpandElementTyped<IntExpand<Marker>>;
}

impl<Marker: 'static> Scalar for IntExpand<Marker> {}
impl<Marker: 'static> CubePrimitive for IntExpand<Marker> {
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

impl<Marker: 'static> From<IntExpand<Marker>> for ConstantValue {
    fn from(val: IntExpand<Marker>) -> Self {
        val.val.into()
    }
}

impl<Marker: 'static> From<IntExpand<Marker>> for Variable {
    fn from(val: IntExpand<Marker>) -> Self {
        // TODO: Fix how we create literal.
        Variable::constant(val.val.into(), cubecl_ir::IntKind::I64)
    }
}

impl<Marker: 'static> From<IntExpand<Marker>> for ExpandElementTyped<IntExpand<Marker>> {
    fn from(value: IntExpand<Marker>) -> Self {
        let var: Variable = value.into();
        ExpandElementTyped::new(ExpandElement::Plain(var))
    }
}

impl<Marker: 'static> IntoRuntime for IntExpand<Marker> {
    fn __expand_runtime_method(self, scope: &mut Scope) -> ExpandElementTyped<Self> {
        ExpandElementTyped::from_lit(scope, self.val)
    }
}

impl<Marker: 'static> Numeric for IntExpand<Marker> {
    fn min_value() -> Self {
        panic!("Can't use min value in comptime with dynamic element type");
    }
    fn max_value() -> Self {
        panic!("Can't use max value in comptime with dynamic element type");
    }
}

impl<Marker: 'static> ExpandElementAssign for IntExpand<Marker> {}

impl<Marker: 'static> ScalarArgSettings for IntExpand<Marker> {
    fn register<R: Runtime>(&self, _launcher: &mut KernelLauncher<R>) {
        panic!("Can't launch `IntExpand` as scalar")
    }
}

impl<Marker: 'static> Remainder for IntExpand<Marker> {}
impl<Marker: 'static> Abs for IntExpand<Marker> {}
impl<Marker: 'static> MulHi for IntExpand<Marker> {}

impl<Marker: 'static> CubeNot for IntExpand<Marker> {}
impl<Marker: 'static> ReverseBits for IntExpand<Marker> {}
impl<Marker: 'static> CountOnes for IntExpand<Marker> {}
impl<Marker: 'static> FindFirstSet for IntExpand<Marker> {}
impl<Marker: 'static> LeadingZeros for IntExpand<Marker> {}
impl<Marker: 'static> TrailingZeros for IntExpand<Marker> {}
impl<Marker: 'static> SaturatingAdd for IntExpand<Marker> {}
impl<Marker: 'static> SaturatingSub for IntExpand<Marker> {}

impl<Marker: 'static> Int for IntExpand<Marker> {
    const BITS: u32 = 32;

    fn new(val: i64) -> Self {
        IntExpand::new(val)
    }
}

impl<Marker: 'static> Zero for IntExpand<Marker> {
    fn zero() -> Self {
        Self::new(0)
    }

    fn is_zero(&self) -> bool {
        self.val == 0
    }
}

impl<Marker: 'static> One for IntExpand<Marker> {
    fn one() -> Self {
        Self::new(1)
    }
}

impl<Marker: 'static> Num for IntExpand<Marker> {
    type FromStrRadixErr = ParseIntError;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Ok(IntExpand::new(i64::from_str_radix(str, radix)?))
    }
}
