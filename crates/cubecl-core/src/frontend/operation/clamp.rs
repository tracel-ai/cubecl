use std::num::NonZero;

use half::{bf16, f16};

use crate::{
    new_ir::{Expanded, Expr, Expression, SquareType},
    prelude::Numeric,
};

pub trait Clamp: PartialOrd + Numeric {
    /// Clamp the input value between the max and min values provided.
    #[allow(unused_variables)]
    fn clamp(self, min_value: Self, max_value: Self) -> Self {
        num_traits::clamp(self, min_value, max_value)
    }
}

pub trait ClampExpand: Expanded
where
    Self::Unexpanded: PartialOrd + Numeric,
{
    fn clamp(
        self,
        min_value: impl Expr<Output = Self::Unexpanded>,
        max_value: impl Expr<Output = Self::Unexpanded>,
    ) -> impl Expr<Output = Self::Unexpanded> {
        ClampExpr::new(self.inner(), min_value, max_value)
    }
}

impl<T: Expanded> ClampExpand for T where T::Unexpanded: PartialOrd + Numeric {}

#[derive(new)]
pub struct ClampExpr<In: Expr, Min: Expr<Output = In::Output>, Max: Expr<Output = In::Output>>
where
    In::Output: Numeric,
{
    pub input: In,
    pub min: Min,
    pub max: Max,
}

impl<In: Expr, Min: Expr<Output = In::Output>, Max: Expr<Output = In::Output>> Expr
    for ClampExpr<In, Min, Max>
where
    In::Output: Numeric,
{
    type Output = In::Output;

    fn expression_untyped(&self) -> Expression {
        Expression::Clamp {
            input: Box::new(self.input.expression_untyped()),
            min: Box::new(self.min.expression_untyped()),
            max: Box::new(self.max.expression_untyped()),
            vectorization: self.vectorization(),
            ty: <In::Output as SquareType>::ir_type(),
        }
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        self.input.vectorization()
    }
}

impl Clamp for f16 {}
impl Clamp for bf16 {}
impl Clamp for f32 {}
impl Clamp for f64 {}
impl Clamp for i32 {}
impl Clamp for i64 {}
impl Clamp for u32 {}
