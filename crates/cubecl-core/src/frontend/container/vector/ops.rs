use core::{marker::PhantomData, ops::Not};
use cubecl_ir::{Bitwise, ConstantValue, ElemType, Instruction, Type, UIntKind, UnaryOperator};
use cubecl_macros::{cube, intrinsic};
use num_traits::{NumCast, One, ToPrimitive, Zero};

use crate::{
    self as cubecl,
    prelude::{
        ArcTan2, InverseSqrt, IsInf, IsNan, Powf, Powi, SaturatingAdd, SaturatingSub, Trunc,
    },
};
use crate::{prelude::*, unexpanded};

use super::Vector;
type VectorExpand<E, N> = NativeExpand<Vector<E, N>>;

impl<P, N: Size> core::ops::Add<Self> for Vector<P, N>
where
    P: Scalar,
    P: core::ops::Add<P, Output = P>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.val + rhs.val)
    }
}

impl<P, N: Size> core::ops::Sub<Self> for Vector<P, N>
where
    P: Scalar,
    P: core::ops::Sub<P, Output = P>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.val - rhs.val)
    }
}

impl<P, N: Size> core::ops::Mul<Self> for Vector<P, N>
where
    P: Scalar,
    P: core::ops::Mul<P, Output = P>,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(self.val * rhs.val)
    }
}

impl<P, N: Size> core::ops::Div<Self> for Vector<P, N>
where
    P: Scalar,
    P: core::ops::Div<P, Output = P>,
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self::new(self.val / rhs.val)
    }
}

impl<P, N: Size> core::ops::AddAssign<Self> for Vector<P, N>
where
    P: Scalar,
    P: core::ops::AddAssign,
{
    fn add_assign(&mut self, rhs: Self) {
        self.val += rhs.val;
    }
}

impl<P, N: Size> core::ops::SubAssign<Self> for Vector<P, N>
where
    P: Scalar,
    P: core::ops::SubAssign,
{
    fn sub_assign(&mut self, rhs: Self) {
        self.val -= rhs.val;
    }
}

impl<P, N: Size> core::ops::DivAssign<Self> for Vector<P, N>
where
    P: Scalar,
    P: core::ops::DivAssign,
{
    fn div_assign(&mut self, rhs: Self) {
        self.val /= rhs.val;
    }
}

impl<P, N: Size> core::ops::MulAssign<Self> for Vector<P, N>
where
    P: Scalar,
    P: core::ops::MulAssign,
{
    fn mul_assign(&mut self, rhs: Self) {
        self.val *= rhs.val;
    }
}

impl<P, N: Size> core::cmp::PartialEq for Vector<P, N>
where
    P: Scalar,
    P: core::cmp::PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.val.eq(&other.val)
    }
}

impl<P, N: Size> core::cmp::PartialOrd for Vector<P, N>
where
    P: Scalar,
    P: core::cmp::PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.val.partial_cmp(&other.val)
    }
}

impl<P, N: Size> core::ops::BitAnd<Self> for Vector<P, N>
where
    P: Scalar,
    P: core::ops::BitAnd<P, Output = P>,
{
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self::new(self.val & rhs.val)
    }
}

impl<P, N: Size> core::ops::BitOr<Self> for Vector<P, N>
where
    P: Scalar,
    P: core::ops::BitOr<P, Output = P>,
{
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self::new(self.val | rhs.val)
    }
}

impl<P, N: Size> core::ops::BitXor<Self> for Vector<P, N>
where
    P: Scalar,
    P: core::ops::BitXor<P, Output = P>,
{
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self::new(self.val ^ rhs.val)
    }
}

impl<P, N: Size> core::ops::Shl<Self> for Vector<P, N>
where
    P: Scalar,
    P: core::ops::Shl<P, Output = P>,
{
    type Output = Self;

    fn shl(self, rhs: Self) -> Self::Output {
        Self::new(self.val << rhs.val)
    }
}

impl<P, N: Size> core::ops::Shr<Self> for Vector<P, N>
where
    P: Scalar,
    P: core::ops::Shr<P, Output = P>,
{
    type Output = Self;

    fn shr(self, rhs: Self) -> Self::Output {
        Self::new(self.val >> rhs.val)
    }
}

impl<P, N: Size> core::ops::BitAndAssign<Self> for Vector<P, N>
where
    P: Scalar,
    P: core::ops::BitAndAssign,
{
    fn bitand_assign(&mut self, rhs: Self) {
        self.val &= rhs.val;
    }
}

impl<P, N: Size> core::ops::BitOrAssign<Self> for Vector<P, N>
where
    P: Scalar,
    P: core::ops::BitOrAssign,
{
    fn bitor_assign(&mut self, rhs: Self) {
        self.val |= rhs.val;
    }
}

impl<P, N: Size> core::ops::BitXorAssign<Self> for Vector<P, N>
where
    P: Scalar,
    P: core::ops::BitXorAssign,
{
    fn bitxor_assign(&mut self, rhs: Self) {
        self.val ^= rhs.val;
    }
}

impl<P, N: Size> core::ops::ShlAssign<Self> for Vector<P, N>
where
    P: Scalar,
    P: core::ops::ShlAssign,
{
    fn shl_assign(&mut self, rhs: Self) {
        self.val <<= rhs.val;
    }
}

impl<P, N: Size> core::ops::ShrAssign<Self> for Vector<P, N>
where
    P: Scalar,
    P: core::ops::ShrAssign,
{
    fn shr_assign(&mut self, rhs: Self) {
        self.val >>= rhs.val;
    }
}

impl<P: Scalar + Abs, N: Size> Abs for Vector<P, N> {
    type AbsElem = P::AbsElem;
}
impl<P: Scalar + Log, N: Size> Log for Vector<P, N> {}
impl<P: Scalar + Log1p, N: Size> Log1p for Vector<P, N> {}
impl<P: Scalar + Expm1, N: Size> Expm1 for Vector<P, N> {}
impl<P: Scalar + Erf, N: Size> Erf for Vector<P, N> {}
impl<P: Scalar + Exp, N: Size> Exp for Vector<P, N> {}
impl<P: Scalar + Powf, N: Size> Powf for Vector<P, N> {}
impl<P: Scalar + Powi<I>, I: Scalar, N: Size> Powi<Vector<I, N>> for Vector<P, N> {}
impl<P: Scalar + Sqrt, N: Size> Sqrt for Vector<P, N> {}
impl<P: Scalar + InverseSqrt, N: Size> InverseSqrt for Vector<P, N> {}
impl<P: Scalar + Cos, N: Size> Cos for Vector<P, N> {}
impl<P: Scalar + Sin, N: Size> Sin for Vector<P, N> {}
impl<P: Scalar + Tan, N: Size> Tan for Vector<P, N> {}
impl<P: Scalar + Tanh, N: Size> Tanh for Vector<P, N> {}
impl<P: Scalar + Sinh, N: Size> Sinh for Vector<P, N> {}
impl<P: Scalar + Cosh, N: Size> Cosh for Vector<P, N> {}
impl<P: Scalar + ArcSin, N: Size> ArcSin for Vector<P, N> {}
impl<P: Scalar + ArcCos, N: Size> ArcCos for Vector<P, N> {}
impl<P: Scalar + ArcTan, N: Size> ArcTan for Vector<P, N> {}
impl<P: Scalar + ArcSinh, N: Size> ArcSinh for Vector<P, N> {}
impl<P: Scalar + ArcCosh, N: Size> ArcCosh for Vector<P, N> {}
impl<P: Scalar + ArcTanh, N: Size> ArcTanh for Vector<P, N> {}
impl<P: Scalar + ArcTan2, N: Size> ArcTan2 for Vector<P, N> {}
impl<P: Scalar + Recip, N: Size> Recip for Vector<P, N> {}
impl<P: Scalar + Remainder, N: Size> Remainder for Vector<P, N> {}
impl<P: Scalar + Round, N: Size> Round for Vector<P, N> {}
impl<P: Scalar + Floor, N: Size> Floor for Vector<P, N> {}
impl<P: Scalar + Ceil, N: Size> Ceil for Vector<P, N> {}
impl<P: Scalar + Trunc, N: Size> Trunc for Vector<P, N> {}
impl<P: Scalar + ReverseBits, N: Size> ReverseBits for Vector<P, N> {}
impl<P: Scalar + CubeNot, N: Size> CubeNot for Vector<P, N> {}
impl<P: Scalar + SaturatingAdd, N: Size> SaturatingAdd for Vector<P, N> {}
impl<P: Scalar + SaturatingSub, N: Size> SaturatingSub for Vector<P, N> {}
impl<P: Scalar + IsNan, N: Size> IsNan for Vector<P, N> {}
impl<P: Scalar + IsInf, N: Size> IsInf for Vector<P, N> {}
impl<P: Scalar + Normalize, N: Size> Normalize for Vector<P, N> {}
impl<P: Scalar + Magnitude, N: Size> Magnitude for Vector<P, N> {}
impl<P: Scalar + VectorSum, N: Size> VectorSum for Vector<P, N> {}
impl<P: Scalar + Degrees, N: Size> Degrees for Vector<P, N> {}
impl<P: Scalar + Radians, N: Size> Radians for Vector<P, N> {}

impl<P: Scalar + Ord, N: Size> Ord for Vector<P, N> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.val.cmp(&other.val)
    }
}

#[cube]
impl<P: CountOnes + Scalar, N: Size> Vector<P, N> {
    pub fn count_ones(self) -> Vector<u32, N> {
        intrinsic!(|scope| {
            let out_item = Type::scalar(ElemType::UInt(UIntKind::U32))
                .with_vector_size(self.expand.ty.vector_size());
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

impl<P: LeadingZeros + Scalar, N: Size> LeadingZeros for Vector<P, N> {}
impl<P: FindFirstSet + Scalar, N: Size> FindFirstSet for Vector<P, N> {}
impl<P: TrailingZeros + Scalar, N: Size> TrailingZeros for Vector<P, N> {}

impl<P: Scalar + NumCast, N: Size> NumCast for Vector<P, N> {
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        let val: P = NumCast::from(n)?;
        Some(Self {
            val,
            _size: PhantomData,
        })
    }
}
impl<P: Scalar + NumCast, N: Size> ToPrimitive for Vector<P, N> {
    fn to_i64(&self) -> Option<i64> {
        self.val.to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.val.to_u64()
    }
}

impl<P: Not<Output = P> + Scalar, N: Size> Not for Vector<P, N> {
    type Output = Self;

    fn not(self) -> Self::Output {
        Vector::new(self.val.not())
    }
}

#[allow(clippy::from_over_into)]
impl<P: Scalar + Into<NativeExpand<P>>, N: Size> Into<NativeExpand<Self>> for Vector<P, N> {
    fn into(self) -> NativeExpand<Self> {
        let elem: NativeExpand<P> = self.val.into();
        elem.expand.into()
    }
}

impl<T: Scalar + Default, N: Size> Default for Vector<T, N> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T: Scalar + IntoRuntime, N: Size> IntoRuntime for Vector<T, N> {
    fn __expand_runtime_method(self, scope: &mut Scope) -> Self::ExpandType {
        let val = self.val.__expand_runtime_method(scope);
        Self::__expand_new(scope, val)
    }
}

impl<T: Scalar + Into<ConstantValue>, N: Size> From<Vector<T, N>> for ConstantValue {
    fn from(value: Vector<T, N>) -> Self {
        value.val.into()
    }
}

impl<T: Scalar + Zero, N: Size> Zero for Vector<T, N> {
    fn zero() -> Self {
        Self::new(T::zero())
    }

    fn is_zero(&self) -> bool {
        self.val.is_zero()
    }
}

impl<T: Scalar + One, N: Size> One for Vector<T, N> {
    fn one() -> Self {
        Self::new(T::one())
    }
}

macro_rules! operation_literal {
    ($lit:ty) => {
        impl<P, N: Size> core::ops::Add<$lit> for Vector<P, N>
        where
            P: Scalar,
        {
            type Output = Self;

            fn add(self, _rhs: $lit) -> Self::Output {
                unexpanded!();
            }
        }

        impl<P, N: Size> core::ops::Sub<$lit> for Vector<P, N>
        where
            P: Scalar,
        {
            type Output = Self;

            fn sub(self, _rhs: $lit) -> Self::Output {
                unexpanded!();
            }
        }

        impl<P, N: Size> core::ops::Mul<$lit> for Vector<P, N>
        where
            P: Scalar,
        {
            type Output = Self;

            fn mul(self, _rhs: $lit) -> Self::Output {
                unexpanded!();
            }
        }

        impl<P, N: Size> core::ops::Div<$lit> for Vector<P, N>
        where
            P: Scalar,
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
