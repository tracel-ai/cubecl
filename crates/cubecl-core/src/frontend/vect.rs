use std::num::NonZero;

use crate::{
    new_ir::{Expand, Expanded, Expr, Expression, SquareType, TensorExpression, Vectorization},
    unexpanded,
};

#[derive(new)]
pub struct VectorizeExpr<T: Expr>
where
    T::Output: SquareType,
{
    pub inner: T,
    pub vectorization: Vectorization,
}

impl<T: Expr> Expr for VectorizeExpr<T>
where
    T::Output: SquareType,
{
    type Output = T::Output;

    fn expression_untyped(&self) -> Expression {
        Expression::Cast {
            from: Box::new(self.inner.expression_untyped()),
            vectorization: self.vectorization(),
            to: <T::Output as SquareType>::ir_type(),
        }
    }

    fn vectorization(&self) -> Vectorization {
        self.vectorization
    }
}

pub fn vectorize<T: SquareType>(_inner: T, _vectorization: u32) -> T {
    unexpanded!()
}

pub fn vectorize_like<T: SquareType, Other: SquareType>(_this: T, _other: &Other) -> T {
    unexpanded!()
}

pub fn vectorization_of<T: SquareType>(_this: &T) -> u32 {
    unexpanded!()
}

pub mod vectorize {
    use super::*;

    pub fn expand<T: SquareType>(
        inner: impl Expr<Output = T>,
        vectorization: u32,
    ) -> impl Expr<Output = T> {
        VectorizeExpr::new(inner, NonZero::new(vectorization as u8))
    }
}

pub mod vectorization_of {
    use super::*;

    pub fn expand<T: SquareType>(this: impl Expr<Output = T>) -> u32 {
        this.vectorization().map(|it| it.get() as u32).unwrap_or(1)
    }
}

pub mod vectorize_like {
    use super::*;

    pub fn expand<T: SquareType, Ref: SquareType>(
        inner: impl Expr<Output = T>,
        other: impl Expr<Output = Ref>,
    ) -> impl Expr<Output = T> {
        VectorizeExpr::new(inner, other.vectorization())
    }
}

#[derive(new)]
pub struct VecIndexExpr<Inner: Expr, Index: Expr<Output = u32>>
where
    Inner::Output: VecIndex,
{
    pub inner: Inner,
    pub index: Index,
}

impl<Inner: Expr, Index: Expr<Output = u32>> Expr for VecIndexExpr<Inner, Index>
where
    Inner::Output: VecIndex,
{
    type Output = Inner::Output;

    fn expression_untyped(&self) -> Expression {
        TensorExpression::Index {
            tensor: Box::new(self.inner.expression_untyped()),
            index: Box::new(self.index.expression_untyped()),
            vectorization: self.vectorization(),
        }
        .into()
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        NonZero::new(1)
    }
}

pub trait VecIndex: Expand {
    fn vec_index(&self, _index: u32) -> Self {
        unexpanded!()
    }
}

pub trait VecIndexMut: VecIndex + Expand {
    fn vec_index_mut(&mut self, _index: u32) -> &mut Self {
        unexpanded!()
    }
}

pub trait VecIndexExpand<Out> {
    fn vec_index(self, index: impl Expr<Output = u32>) -> impl Expr<Output = Out>;
}
pub trait VecIndexMutExpand<Out> {
    fn vec_index_mut(self, index: impl Expr<Output = u32>) -> impl Expr<Output = Out>;
}

impl<Expansion: Expanded> VecIndexExpand<Expansion::Unexpanded> for Expansion
where
    Expansion::Unexpanded: VecIndex,
{
    fn vec_index(
        self,
        index: impl Expr<Output = u32>,
    ) -> impl Expr<Output = Expansion::Unexpanded> {
        VecIndexExpr::new(self.inner(), index)
    }
}

impl<T: Expanded> VecIndexMutExpand<T::Unexpanded> for T
where
    T::Unexpanded: VecIndexMut,
{
    fn vec_index_mut(self, index: impl Expr<Output = u32>) -> impl Expr<Output = T::Unexpanded> {
        VecIndexExpr::new(self.inner(), index)
    }
}
