use crate::{
    new_ir::{largest_common_vectorization, Expr, Expression, SquareType, Vectorization},
    prelude::Numeric,
};

/// Fused multiply-add `A*B+C`.
#[allow(unused_variables)]
pub fn fma<C: Numeric>(a: C, b: C, c: C) -> C {
    a + b * c
}

pub mod fma {
    use crate::{new_ir::Expr, prelude::Numeric};

    use super::FmaExpr;

    pub fn expand<C: Numeric>(
        a: impl Expr<Output = C>,
        b: impl Expr<Output = C>,
        c: impl Expr<Output = C>,
    ) -> impl Expr<Output = C> {
        FmaExpr::new(a, b, c)
    }
}

#[derive(new)]
pub struct FmaExpr<A: Expr, B: Expr<Output = A::Output>, C: Expr<Output = A::Output>>
where
    A::Output: Numeric,
{
    pub a: A,
    pub b: B,
    pub c: C,
}

impl<A: Expr, B: Expr<Output = A::Output>, C: Expr<Output = A::Output>> Expr for FmaExpr<A, B, C>
where
    A::Output: Numeric,
{
    type Output = A::Output;

    fn expression_untyped(&self) -> Expression {
        Expression::Fma {
            a: Box::new(self.a.expression_untyped()),
            b: Box::new(self.b.expression_untyped()),
            c: Box::new(self.c.expression_untyped()),
            ty: <A::Output as SquareType>::ir_type(),
            vectorization: self.vectorization(),
        }
    }

    fn vectorization(&self) -> Vectorization {
        let a_b = largest_common_vectorization(self.a.vectorization(), self.b.vectorization());
        largest_common_vectorization(a_b, self.c.vectorization())
    }
}
