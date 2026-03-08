use core::{
    marker::PhantomData,
    ops::{Not, Rem, RemAssign},
};
use cubecl_ir::{Bitwise, ConstantValue, ElemType, Instruction, Type, UIntKind, UnaryOperator};
use cubecl_macros::{cube, intrinsic};
use num_traits::{Num, NumCast, One, ToPrimitive, Zero};

use crate::{
    self as cubecl,
    prelude::{
        ArcTan2, InverseSqrt, IsInf, IsNan, Powf, Powi, SaturatingAdd, SaturatingSub, Trunc,
    },
};
use crate::{prelude::*, unexpanded};

use super::Line;
type LineExpand<E, N> = ExpandElementTyped<Line<E, N>>;

impl<P, N: Size> core::ops::Add<Self> for Line<P, N>
where
    P: CubePrimitive,
    P: core::ops::Add<P, Output = P>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.val + rhs.val)
    }
}

impl<P, N: Size> core::ops::Sub<Self> for Line<P, N>
where
    P: CubePrimitive,
    P: core::ops::Sub<P, Output = P>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.val - rhs.val)
    }
}

impl<P, N: Size> core::ops::Mul<Self> for Line<P, N>
where
    P: CubePrimitive,
    P: core::ops::Mul<P, Output = P>,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(self.val * rhs.val)
    }
}

impl<P, N: Size> core::ops::Div<Self> for Line<P, N>
where
    P: CubePrimitive,
    P: core::ops::Div<P, Output = P>,
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self::new(self.val / rhs.val)
    }
}

impl<P, N: Size> core::ops::AddAssign<Self> for Line<P, N>
where
    P: CubePrimitive,
    P: core::ops::AddAssign,
{
    fn add_assign(&mut self, rhs: Self) {
        self.val += rhs.val;
    }
}

impl<P, N: Size> core::ops::SubAssign<Self> for Line<P, N>
where
    P: CubePrimitive,
    P: core::ops::SubAssign,
{
    fn sub_assign(&mut self, rhs: Self) {
        self.val -= rhs.val;
    }
}

impl<P, N: Size> core::ops::DivAssign<Self> for Line<P, N>
where
    P: CubePrimitive,
    P: core::ops::DivAssign,
{
    fn div_assign(&mut self, rhs: Self) {
        self.val /= rhs.val;
    }
}

impl<P, N: Size> core::ops::MulAssign<Self> for Line<P, N>
where
    P: CubePrimitive,
    P: core::ops::MulAssign,
{
    fn mul_assign(&mut self, rhs: Self) {
        self.val *= rhs.val;
    }
}

impl<P, N: Size> core::cmp::PartialEq for Line<P, N>
where
    P: CubePrimitive,
    P: core::cmp::PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.val.eq(&other.val)
    }
}

impl<P, N: Size> core::cmp::PartialOrd for Line<P, N>
where
    P: CubePrimitive,
    P: core::cmp::PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.val.partial_cmp(&other.val)
    }
}

impl<P, N: Size> core::ops::BitAnd<Self> for Line<P, N>
where
    P: CubePrimitive,
    P: core::ops::BitAnd<P, Output = P>,
{
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self::new(self.val & rhs.val)
    }
}

impl<P, N: Size> core::ops::BitOr<Self> for Line<P, N>
where
    P: CubePrimitive,
    P: core::ops::BitOr<P, Output = P>,
{
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self::new(self.val | rhs.val)
    }
}

impl<P, N: Size> core::ops::BitXor<Self> for Line<P, N>
where
    P: CubePrimitive,
    P: core::ops::BitXor<P, Output = P>,
{
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self::new(self.val ^ rhs.val)
    }
}

impl<P, N: Size> core::ops::Shl<Self> for Line<P, N>
where
    P: CubePrimitive,
    P: core::ops::Shl<P, Output = P>,
{
    type Output = Self;

    fn shl(self, rhs: Self) -> Self::Output {
        Self::new(self.val << rhs.val)
    }
}

impl<P, N: Size> core::ops::Shr<Self> for Line<P, N>
where
    P: CubePrimitive,
    P: core::ops::Shr<P, Output = P>,
{
    type Output = Self;

    fn shr(self, rhs: Self) -> Self::Output {
        Self::new(self.val >> rhs.val)
    }
}

impl<P, N: Size> core::ops::BitAndAssign<Self> for Line<P, N>
where
    P: CubePrimitive,
    P: core::ops::BitAndAssign,
{
    fn bitand_assign(&mut self, rhs: Self) {
        self.val &= rhs.val;
    }
}

impl<P, N: Size> core::ops::BitOrAssign<Self> for Line<P, N>
where
    P: CubePrimitive,
    P: core::ops::BitOrAssign,
{
    fn bitor_assign(&mut self, rhs: Self) {
        self.val |= rhs.val;
    }
}

impl<P, N: Size> core::ops::BitXorAssign<Self> for Line<P, N>
where
    P: CubePrimitive,
    P: core::ops::BitXorAssign,
{
    fn bitxor_assign(&mut self, rhs: Self) {
        self.val ^= rhs.val;
    }
}

impl<P, N: Size> core::ops::ShlAssign<Self> for Line<P, N>
where
    P: CubePrimitive,
    P: core::ops::ShlAssign,
{
    fn shl_assign(&mut self, rhs: Self) {
        self.val <<= rhs.val;
    }
}

impl<P, N: Size> core::ops::ShrAssign<Self> for Line<P, N>
where
    P: CubePrimitive,
    P: core::ops::ShrAssign,
{
    fn shr_assign(&mut self, rhs: Self) {
        self.val >>= rhs.val;
    }
}

impl<P: CubePrimitive + Abs, N: Size> Abs for Line<P, N> {}
impl<P: CubePrimitive + Log, N: Size> Log for Line<P, N> {}
impl<P: CubePrimitive + Log1p, N: Size> Log1p for Line<P, N> {}
impl<P: CubePrimitive + Erf, N: Size> Erf for Line<P, N> {}
impl<P: CubePrimitive + Exp, N: Size> Exp for Line<P, N> {}
impl<P: CubePrimitive + Powf, N: Size> Powf for Line<P, N> {}
impl<P: CubePrimitive + Powi<I>, I: CubePrimitive, N: Size> Powi<Line<I, N>> for Line<P, N> {}
impl<P: CubePrimitive + Sqrt, N: Size> Sqrt for Line<P, N> {}
impl<P: CubePrimitive + InverseSqrt, N: Size> InverseSqrt for Line<P, N> {}
impl<P: CubePrimitive + Cos, N: Size> Cos for Line<P, N> {}
impl<P: CubePrimitive + Sin, N: Size> Sin for Line<P, N> {}
impl<P: CubePrimitive + Tan, N: Size> Tan for Line<P, N> {}
impl<P: CubePrimitive + Tanh, N: Size> Tanh for Line<P, N> {}
impl<P: CubePrimitive + Sinh, N: Size> Sinh for Line<P, N> {}
impl<P: CubePrimitive + Cosh, N: Size> Cosh for Line<P, N> {}
impl<P: CubePrimitive + ArcSin, N: Size> ArcSin for Line<P, N> {}
impl<P: CubePrimitive + ArcCos, N: Size> ArcCos for Line<P, N> {}
impl<P: CubePrimitive + ArcTan, N: Size> ArcTan for Line<P, N> {}
impl<P: CubePrimitive + ArcSinh, N: Size> ArcSinh for Line<P, N> {}
impl<P: CubePrimitive + ArcCosh, N: Size> ArcCosh for Line<P, N> {}
impl<P: CubePrimitive + ArcTanh, N: Size> ArcTanh for Line<P, N> {}
impl<P: CubePrimitive + ArcTan2, N: Size> ArcTan2 for Line<P, N> {}
impl<P: CubePrimitive + Recip, N: Size> Recip for Line<P, N> {}
impl<P: CubePrimitive + Remainder, N: Size> Remainder for Line<P, N> {}
impl<P: CubePrimitive + Round, N: Size> Round for Line<P, N> {}
impl<P: CubePrimitive + Floor, N: Size> Floor for Line<P, N> {}
impl<P: CubePrimitive + Ceil, N: Size> Ceil for Line<P, N> {}
impl<P: CubePrimitive + Trunc, N: Size> Trunc for Line<P, N> {}
impl<P: CubePrimitive + ReverseBits, N: Size> ReverseBits for Line<P, N> {}
impl<P: CubePrimitive + CubeNot, N: Size> CubeNot for Line<P, N> {}
impl<P: CubePrimitive + SaturatingAdd, N: Size> SaturatingAdd for Line<P, N> {}
impl<P: CubePrimitive + SaturatingSub, N: Size> SaturatingSub for Line<P, N> {}
impl<P: CubePrimitive + IsNan, N: Size> IsNan for Line<P, N> {}
impl<P: CubePrimitive + IsInf, N: Size> IsInf for Line<P, N> {}

impl<P: CubePrimitive + Ord, N: Size> Ord for Line<P, N> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.val.cmp(&other.val)
    }
}

#[cube]
impl<P: CountOnes + CubePrimitive, N: Size> Line<P, N> {
    pub fn count_ones(self) -> Line<u32, N> {
        intrinsic!(|scope| {
            let out_item =
                Type::scalar(ElemType::UInt(UIntKind::U32)).line(self.expand.ty.line_size());
            let out = scope.create_local(out_item);
            scope.register(Instruction::new(
                Bitwise::CountOnes(UnaryOperator {
                    input: *self.expand,
                }),
                *out,
            ));
            out.into()
        })
    }
}

#[cube]
impl<P: LeadingZeros + CubePrimitive, N: Size> Line<P, N> {
    pub fn leading_zeros(self) -> Line<u32, N> {
        intrinsic!(|scope| {
            let out_item =
                Type::scalar(ElemType::UInt(UIntKind::U32)).line(self.expand.ty.line_size());
            let out = scope.create_local(out_item);
            scope.register(Instruction::new(
                Bitwise::LeadingZeros(UnaryOperator {
                    input: *self.expand,
                }),
                *out,
            ));
            out.into()
        })
    }
}

#[cube]
impl<P: FindFirstSet + CubePrimitive, N: Size> Line<P, N> {
    pub fn find_first_set(self) -> Line<u32, N> {
        intrinsic!(|scope| {
            let out_item =
                Type::scalar(ElemType::UInt(UIntKind::U32)).line(self.expand.ty.line_size());
            let out = scope.create_local(out_item);
            scope.register(Instruction::new(
                Bitwise::FindFirstSet(UnaryOperator {
                    input: *self.expand,
                }),
                *out,
            ));
            out.into()
        })
    }
}

impl<P: CubePrimitive + NumCast, N: Size> NumCast for Line<P, N> {
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        let val: P = NumCast::from(n)?;
        Some(Self {
            val,
            _size: PhantomData,
        })
    }
}
impl<P: CubePrimitive + NumCast, N: Size> ToPrimitive for Line<P, N> {
    fn to_i64(&self) -> Option<i64> {
        self.val.to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.val.to_u64()
    }
}

impl<P: Not<Output = P> + CubePrimitive, N: Size> Not for Line<P, N> {
    type Output = Self;

    fn not(self) -> Self::Output {
        Line::new(self.val.not())
    }
}

#[allow(clippy::from_over_into)]
impl<P: CubePrimitive + Into<ExpandElementTyped<P>>, N: Size> Into<ExpandElementTyped<Self>>
    for Line<P, N>
{
    fn into(self) -> ExpandElementTyped<Self> {
        let elem: ExpandElementTyped<P> = self.val.into();
        elem.expand.into()
    }
}

impl<T: CubePrimitive + Default, N: Size> Default for Line<T, N> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T: Numeric, N: Size> Numeric for Line<T, N> {
    fn min_value() -> Self {
        Self::new(T::min_value())
    }

    fn max_value() -> Self {
        Self::new(T::max_value())
    }
}

impl<T: CubePrimitive + Rem<Output = T>, N: Size> Rem for Line<T, N> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        Self::new(self.val % rhs.val)
    }
}

impl<T: CubePrimitive + RemAssign, N: Size> RemAssign for Line<T, N> {
    fn rem_assign(&mut self, rhs: Self) {
        self.val %= rhs.val;
    }
}

impl<T: CubePrimitive + IntoRuntime, N: Size> IntoRuntime for Line<T, N> {
    fn __expand_runtime_method(self, scope: &mut Scope) -> Self::ExpandType {
        let val = self.val.__expand_runtime_method(scope);
        Self::__expand_new(scope, val)
    }
}

impl<T: CubePrimitive + Into<ConstantValue>, N: Size> From<Line<T, N>> for ConstantValue {
    fn from(value: Line<T, N>) -> Self {
        value.val.into()
    }
}

impl<T: CubePrimitive + Num, N: Size> Num for Line<T, N> {
    type FromStrRadixErr = T::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Ok(Self::new(T::from_str_radix(str, radix)?))
    }
}

impl<T: CubePrimitive + Zero, N: Size> Zero for Line<T, N> {
    fn zero() -> Self {
        Self::new(T::zero())
    }

    fn is_zero(&self) -> bool {
        self.val.is_zero()
    }
}

impl<T: CubePrimitive + One, N: Size> One for Line<T, N> {
    fn one() -> Self {
        Self::new(T::one())
    }
}

macro_rules! operation_literal {
    ($lit:ty) => {
        impl<P, N: Size> core::ops::Add<$lit> for Line<P, N>
        where
            P: CubePrimitive,
        {
            type Output = Self;

            fn add(self, _rhs: $lit) -> Self::Output {
                unexpanded!();
            }
        }

        impl<P, N: Size> core::ops::Sub<$lit> for Line<P, N>
        where
            P: CubePrimitive,
        {
            type Output = Self;

            fn sub(self, _rhs: $lit) -> Self::Output {
                unexpanded!();
            }
        }

        impl<P, N: Size> core::ops::Mul<$lit> for Line<P, N>
        where
            P: CubePrimitive,
        {
            type Output = Self;

            fn mul(self, _rhs: $lit) -> Self::Output {
                unexpanded!();
            }
        }

        impl<P, N: Size> core::ops::Div<$lit> for Line<P, N>
        where
            P: CubePrimitive,
        {
            type Output = Self;

            fn div(self, _rhs: $lit) -> Self::Output {
                unexpanded!();
            }
        }
    };
}

operation_literal!(f32);
operation_literal!(f64);
operation_literal!(usize);
operation_literal!(i32);
operation_literal!(i64);
