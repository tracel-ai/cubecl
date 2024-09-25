use std::num::NonZero;

use num_traits::{NumCast, ToPrimitive};

use crate::{
    ir::{ConstantScalarValue, Item},
    prelude::{assign, Abs, Clamp, CubeContext, CubeIndex, CubeIndexMut, Erf, Log, Max, Min},
    unexpanded,
};

use super::{CubePrimitive, CubeType, ExpandElementBaseInit, ExpandElementTyped, IntoRuntime};

/// A contiguous list of elements that supports auto-vectorization.
#[derive(Clone, Copy, Eq)]
pub struct Line<P: CubePrimitive> {
    // Comptime lines only support 1 element.
    val: P,
}

impl<P: CubePrimitive> CubeType for Line<P> {
    type ExpandType = ExpandElementTyped<Self>;
}

impl<P: CubePrimitive> Line<P> {
    pub fn new(val: P) -> Self {
        Self { val }
    }

    pub fn __expand_new(
        _context: &mut CubeContext,
        val: P::ExpandType,
    ) -> ExpandElementTyped<Self> {
        let elem: ExpandElementTyped<P> = val.into();
        elem.expand.into()
    }
}

impl<P: CubePrimitive> ExpandElementTyped<Line<P>> {
    pub fn __expand_fill_method(
        &self,
        context: &mut CubeContext,
        value: ExpandElementTyped<P>,
    ) -> Self {
        let length = self.expand.item().vectorization;
        let output = context.create_local_binding(Item::vectorized(P::as_elem(), length));

        assign::expand::<P>(context, value.into(), output.clone().into());

        output.into()
    }
}

impl<P: CubePrimitive + Into<ExpandElementTyped<P>>> Line<P> {
    /// Create an empty line of the given length.
    #[allow(unused_variables)]
    pub fn empty(length: u32) -> Self {
        unexpanded!()
    }

    #[allow(unused_variables)]
    pub fn fill(mut self, value: P) -> Self {
        self.val = value;
        self
    }

    /// Get the length of the line.
    pub fn len(&self) -> u32 {
        unexpanded!()
    }

    pub fn __expand_len(context: &mut CubeContext, element: ExpandElementTyped<P>) -> u32 {
        element.__expand_vectorization_factor_method(context)
    }

    pub fn __expand_fill(
        context: &mut CubeContext,
        line: ExpandElementTyped<Self>,
        value: ExpandElementTyped<P>,
    ) -> ExpandElementTyped<Self> {
        line.__expand_fill_method(context, value)
    }

    pub fn __expand_empty(
        context: &mut CubeContext,
        length: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<Self> {
        let length = match length.expand.as_const() {
            Some(val) => match val {
                ConstantScalarValue::Int(val, _) => NonZero::new(val)
                    .map(|val| val.get() as u8)
                    .map(|val| NonZero::new(val).unwrap()),
                ConstantScalarValue::Float(val, _) => NonZero::new(val as i64)
                    .map(|val| val.get() as u8)
                    .map(|val| NonZero::new(val).unwrap()),
                ConstantScalarValue::UInt(val) => NonZero::new(val as u8),
                ConstantScalarValue::Bool(_) => None,
            },
            None => None,
        };
        context
            .create_local_variable(Item::vectorized(Self::as_elem(), length))
            .into()
    }
}

// impl<P: CubePrimitive> LaunchArg for Line<P> {
//     type RuntimeArg<'a, R: crate::Runtime> = ();
//
//     fn compilation_arg<R: crate::Runtime>(
//         runtime_arg: &Self::RuntimeArg<'_, R>,
//     ) -> Self::CompilationArg {
//         todo!()
//     }
// }
//
// impl<P: CubePrimitive> LaunchArgExpand for Line<P> {
//     type CompilationArg = ();
//
//     fn expand(
//         arg: &Self::CompilationArg,
//         builder: &mut crate::prelude::KernelBuilder,
//     ) -> <Self as CubeType>::ExpandType {
//         todo!()
//     }
// }

impl<P: CubePrimitive> ExpandElementBaseInit for Line<P> {
    fn init_elem(
        context: &mut crate::prelude::CubeContext,
        elem: super::ExpandElement,
    ) -> super::ExpandElement {
        P::init_elem(context, elem)
    }
}

impl<P: CubePrimitive> IntoRuntime for Line<P> {
    fn __expand_runtime_method(
        self,
        context: &mut crate::prelude::CubeContext,
    ) -> Self::ExpandType {
        self.val.__expand_runtime_method(context).expand.into()
    }
}

impl<P: CubePrimitive> CubePrimitive for Line<P> {
    fn as_elem() -> crate::ir::Elem {
        P::as_elem()
    }
}

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
impl<P: CubePrimitive + Erf> Erf for Line<P> {}

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
