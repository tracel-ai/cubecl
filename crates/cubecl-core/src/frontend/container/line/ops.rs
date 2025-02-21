use cubecl_ir::{Bitwise, Elem, Instruction, Scope, UIntKind, UnaryOperator};
use num_traits::{NumCast, ToPrimitive};

use crate::{
    frontend::{
        Abs, Ceil, Clamp, Cos, CubeIndex, CubeIndexMut, CubePrimitive, Erf, Exp,
        ExpandElementTyped, Floor, Log, Log1p, Max, Min, Powf, Recip, Remainder, Round, Sin, Sqrt,
        Tanh,
    },
    prelude::{BitwiseNot, CountOnes, FindFirstSet, LeadingZeros, ReverseBits},
    unexpanded,
};

use super::Line;

impl<P> core::ops::Add<Self> for Line<P>
where
    P: CubePrimitive,
    P: core::ops::Add<P, Output = P>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.val + rhs.val)
    }
}

impl<P> core::ops::Sub<Self> for Line<P>
where
    P: CubePrimitive,
    P: core::ops::Sub<P, Output = P>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.val - rhs.val)
    }
}

impl<P> core::ops::Mul<Self> for Line<P>
where
    P: CubePrimitive,
    P: core::ops::Mul<P, Output = P>,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(self.val * rhs.val)
    }
}

impl<P> core::ops::Div<Self> for Line<P>
where
    P: CubePrimitive,
    P: core::ops::Div<P, Output = P>,
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self::new(self.val / rhs.val)
    }
}

impl<P> core::ops::AddAssign<Self> for Line<P>
where
    P: CubePrimitive,
    P: core::ops::AddAssign,
{
    fn add_assign(&mut self, rhs: Self) {
        self.val += rhs.val;
    }
}

impl<P> core::ops::SubAssign<Self> for Line<P>
where
    P: CubePrimitive,
    P: core::ops::SubAssign,
{
    fn sub_assign(&mut self, rhs: Self) {
        self.val -= rhs.val;
    }
}

impl<P> core::ops::DivAssign<Self> for Line<P>
where
    P: CubePrimitive,
    P: core::ops::DivAssign,
{
    fn div_assign(&mut self, rhs: Self) {
        self.val /= rhs.val;
    }
}

impl<P> core::ops::MulAssign<Self> for Line<P>
where
    P: CubePrimitive,
    P: core::ops::MulAssign,
{
    fn mul_assign(&mut self, rhs: Self) {
        self.val *= rhs.val;
    }
}

impl<P> core::cmp::PartialEq for Line<P>
where
    P: CubePrimitive,
    P: core::cmp::PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.val.eq(&other.val)
    }
}

impl<P> core::cmp::PartialOrd for Line<P>
where
    P: CubePrimitive,
    P: core::cmp::PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.val.partial_cmp(&other.val)
    }
}

impl<P> core::ops::BitAnd<Self> for Line<P>
where
    P: CubePrimitive,
    P: core::ops::BitAnd<P, Output = P>,
{
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self::new(self.val & rhs.val)
    }
}

impl<P> core::ops::BitOr<Self> for Line<P>
where
    P: CubePrimitive,
    P: core::ops::BitOr<P, Output = P>,
{
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self::new(self.val | rhs.val)
    }
}

impl<P> core::ops::BitXor<Self> for Line<P>
where
    P: CubePrimitive,
    P: core::ops::BitXor<P, Output = P>,
{
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self::new(self.val ^ rhs.val)
    }
}

impl<P> core::ops::Shl<Self> for Line<P>
where
    P: CubePrimitive,
    P: core::ops::Shl<P, Output = P>,
{
    type Output = Self;

    fn shl(self, rhs: Self) -> Self::Output {
        Self::new(self.val << rhs.val)
    }
}

impl<P> core::ops::Shr<Self> for Line<P>
where
    P: CubePrimitive,
    P: core::ops::Shr<P, Output = P>,
{
    type Output = Self;

    fn shr(self, rhs: Self) -> Self::Output {
        Self::new(self.val >> rhs.val)
    }
}

impl<P> core::ops::BitAndAssign<Self> for Line<P>
where
    P: CubePrimitive,
    P: core::ops::BitAndAssign,
{
    fn bitand_assign(&mut self, rhs: Self) {
        self.val &= rhs.val;
    }
}

impl<P> core::ops::BitOrAssign<Self> for Line<P>
where
    P: CubePrimitive,
    P: core::ops::BitOrAssign,
{
    fn bitor_assign(&mut self, rhs: Self) {
        self.val |= rhs.val;
    }
}

impl<P> core::ops::BitXorAssign<Self> for Line<P>
where
    P: CubePrimitive,
    P: core::ops::BitXorAssign,
{
    fn bitxor_assign(&mut self, rhs: Self) {
        self.val ^= rhs.val;
    }
}

impl<P> core::ops::ShlAssign<Self> for Line<P>
where
    P: CubePrimitive,
    P: core::ops::ShlAssign,
{
    fn shl_assign(&mut self, rhs: Self) {
        self.val <<= rhs.val;
    }
}

impl<P> core::ops::ShrAssign<Self> for Line<P>
where
    P: CubePrimitive,
    P: core::ops::ShrAssign,
{
    fn shr_assign(&mut self, rhs: Self) {
        self.val >>= rhs.val;
    }
}

impl<P: CubePrimitive + Abs> Abs for Line<P> {}
impl<P: CubePrimitive + Max> Max for Line<P> {}
impl<P: CubePrimitive + Min> Min for Line<P> {}
impl<P: CubePrimitive + Clamp> Clamp for Line<P> {}
impl<P: CubePrimitive + Log> Log for Line<P> {}
impl<P: CubePrimitive + Log1p> Log1p for Line<P> {}
impl<P: CubePrimitive + Erf> Erf for Line<P> {}
impl<P: CubePrimitive + Exp> Exp for Line<P> {}
impl<P: CubePrimitive + Powf> Powf for Line<P> {}
impl<P: CubePrimitive + Sqrt> Sqrt for Line<P> {}
impl<P: CubePrimitive + Cos> Cos for Line<P> {}
impl<P: CubePrimitive + Sin> Sin for Line<P> {}
impl<P: CubePrimitive + Tanh> Tanh for Line<P> {}
impl<P: CubePrimitive + Recip> Recip for Line<P> {}
impl<P: CubePrimitive + Remainder> Remainder for Line<P> {}
impl<P: CubePrimitive + Round> Round for Line<P> {}
impl<P: CubePrimitive + Floor> Floor for Line<P> {}
impl<P: CubePrimitive + Ceil> Ceil for Line<P> {}
impl<P: CubePrimitive + ReverseBits> ReverseBits for Line<P> {}
impl<P: CubePrimitive + BitwiseNot> BitwiseNot for Line<P> {}

impl<P: CountOnes> Line<P> {
    pub fn count_ones(self) -> Line<u32> {
        unexpanded!()
    }

    pub fn __expand_count_ones(
        scope: &mut Scope,
        value: ExpandElementTyped<Self>,
    ) -> ExpandElementTyped<Line<u32>> {
        value.__expand_count_ones_method(scope)
    }
}

impl<P: CountOnes> ExpandElementTyped<Line<P>> {
    pub fn __expand_count_ones_method(self, scope: &mut Scope) -> ExpandElementTyped<Line<u32>> {
        let mut out_item = self.expand.item;
        out_item.elem = Elem::UInt(UIntKind::U32);
        let out = scope.create_local(out_item);
        scope.register(Instruction::new(
            Bitwise::CountOnes(UnaryOperator {
                input: *self.expand,
            }),
            *out,
        ));
        out.into()
    }
}

impl<P: LeadingZeros> Line<P> {
    pub fn leading_zeros(self) -> Line<u32> {
        unexpanded!()
    }

    pub fn __expand_leading_zeros(
        scope: &mut Scope,
        value: ExpandElementTyped<Self>,
    ) -> ExpandElementTyped<Line<u32>> {
        value.__expand_leading_zeros_method(scope)
    }
}

impl<P: LeadingZeros> ExpandElementTyped<Line<P>> {
    pub fn __expand_leading_zeros_method(self, scope: &mut Scope) -> ExpandElementTyped<Line<u32>> {
        let mut out_item = self.expand.item;
        out_item.elem = Elem::UInt(UIntKind::U32);
        let out = scope.create_local(out_item);
        scope.register(Instruction::new(
            Bitwise::LeadingZeros(UnaryOperator {
                input: *self.expand,
            }),
            *out,
        ));
        out.into()
    }
}

impl<P: FindFirstSet> Line<P> {
    pub fn find_first_set(self) -> Line<u32> {
        unexpanded!()
    }

    pub fn __expand_find_first_set(
        scope: &mut Scope,
        value: ExpandElementTyped<Self>,
    ) -> ExpandElementTyped<Line<u32>> {
        value.__expand_find_first_set_method(scope)
    }
}

impl<P: FindFirstSet> ExpandElementTyped<Line<P>> {
    pub fn __expand_find_first_set_method(
        self,
        scope: &mut Scope,
    ) -> ExpandElementTyped<Line<u32>> {
        let mut out_item = self.expand.item;
        out_item.elem = Elem::UInt(UIntKind::U32);
        let out = scope.create_local(out_item);
        scope.register(Instruction::new(
            Bitwise::FindFirstSet(UnaryOperator {
                input: *self.expand,
            }),
            *out,
        ));
        out.into()
    }
}

impl<P: CubePrimitive + NumCast> NumCast for Line<P> {
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        let val: P = NumCast::from(n)?;
        Some(Self { val })
    }
}
impl<P: CubePrimitive + NumCast> ToPrimitive for Line<P> {
    fn to_i64(&self) -> Option<i64> {
        self.val.to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.val.to_u64()
    }
}

impl<P> CubeIndex<u32> for Line<P>
where
    P: CubePrimitive,
{
    type Output = P;

    fn cube_idx(&self, _i: u32) -> &Self::Output {
        unexpanded!()
    }
}

impl<P> CubeIndexMut<u32> for Line<P>
where
    P: CubePrimitive,
{
    fn cube_idx_mut(&mut self, _i: u32) -> &mut Self::Output {
        unexpanded!()
    }
}

impl<P> CubeIndex<ExpandElementTyped<u32>> for Line<P>
where
    P: CubePrimitive,
{
    type Output = P;

    fn cube_idx(&self, _i: ExpandElementTyped<u32>) -> &Self::Output {
        unexpanded!()
    }
}

impl<P> CubeIndexMut<ExpandElementTyped<u32>> for Line<P>
where
    P: CubePrimitive,
{
    fn cube_idx_mut(&mut self, _i: ExpandElementTyped<u32>) -> &mut Self::Output {
        unexpanded!()
    }
}

#[allow(clippy::from_over_into)]
impl<P: CubePrimitive + Into<ExpandElementTyped<P>>> Into<ExpandElementTyped<Self>> for Line<P> {
    fn into(self) -> ExpandElementTyped<Self> {
        let elem: ExpandElementTyped<P> = self.val.into();
        elem.expand.into()
    }
}

macro_rules! operation_literal {
    ($lit:ty) => {
        impl<P> core::ops::Add<$lit> for Line<P>
        where
            P: CubePrimitive,
        {
            type Output = Self;

            fn add(self, _rhs: $lit) -> Self::Output {
                unexpanded!();
            }
        }

        impl<P> core::ops::Sub<$lit> for Line<P>
        where
            P: CubePrimitive,
        {
            type Output = Self;

            fn sub(self, _rhs: $lit) -> Self::Output {
                unexpanded!();
            }
        }

        impl<P> core::ops::Mul<$lit> for Line<P>
        where
            P: CubePrimitive,
        {
            type Output = Self;

            fn mul(self, _rhs: $lit) -> Self::Output {
                unexpanded!();
            }
        }

        impl<P> core::ops::Div<$lit> for Line<P>
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
