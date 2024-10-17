use num_traits::{NumCast, ToPrimitive};

use crate::{
    frontend::{
        Abs, Ceil, Clamp, Cos, CubeIndex, CubeIndexMut, CubePrimitive, Erf, Exp,
        ExpandElementTyped, Floor, Log, Log1p, Max, Min, Powf, Recip, Remainder, Round, Sin, Sqrt,
        Tanh,
    },
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
    P: CubePrimitive + CubeIndex<u32, Output = P>,
{
    type Output = P;

    fn cube_idx(&self, _i: u32) -> &Self::Output {
        unexpanded!()
    }
}

impl<P> CubeIndexMut<u32> for Line<P>
where
    P: CubePrimitive + CubeIndexMut<u32, Output = P>,
{
    fn cube_idx_mut(&mut self, _i: u32) -> &mut Self::Output {
        unexpanded!()
    }
}

impl<P> CubeIndex<ExpandElementTyped<u32>> for Line<P>
where
    P: CubePrimitive + CubeIndex<ExpandElementTyped<u32>, Output = P>,
{
    type Output = P;

    fn cube_idx(&self, _i: ExpandElementTyped<u32>) -> &Self::Output {
        unexpanded!()
    }
}

impl<P> CubeIndexMut<ExpandElementTyped<u32>> for Line<P>
where
    P: CubePrimitive + CubeIndexMut<ExpandElementTyped<u32>, Output = P>,
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
