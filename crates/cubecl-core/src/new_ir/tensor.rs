use std::{marker::PhantomData, ops::Index};

use super::{Elem, Expr, Expression, Integer, SquareType};

#[derive(Clone, Debug, PartialEq)]
pub enum TensorExpression {
    Stride {
        tensor: Box<Expression>,
        dim: Box<Expression>,
    },
    Shape {
        tensor: Box<Expression>,
        dim: Box<Expression>,
    },
    Length {
        tensor: Box<Expression>,
    },
    Rank {
        tensor: Box<Expression>,
    },
    Index {
        tensor: Box<Expression>,
        index: Box<Expression>,
    },
}

impl TensorExpression {
    pub fn ir_type(&self) -> Elem {
        match self {
            TensorExpression::Stride { dim, .. } => dim.ir_type(),
            TensorExpression::Shape { dim, .. } => dim.ir_type(),
            TensorExpression::Length { .. } => Elem::UInt,
            TensorExpression::Rank { .. } => Elem::UInt,
            TensorExpression::Index { tensor, .. } => tensor.ir_type(),
        }
    }
}

pub trait Strided {}

#[derive(new)]
pub struct Stride<Tensor: Expr, Dim: Expr>
where
    Tensor::Output: Strided,
    Dim::Output: Integer,
{
    pub tensor: Tensor,
    pub dim: Dim,
}

impl<Tensor: Expr, Dim: Expr> Expr for Stride<Tensor, Dim>
where
    Tensor::Output: Strided,
    Dim::Output: Integer,
{
    type Output = Dim::Output;

    fn expression_untyped(&self) -> super::Expression {
        Expression::Tensor(TensorExpression::Stride {
            tensor: Box::new(self.tensor.expression_untyped()),
            dim: Box::new(self.dim.expression_untyped()),
        })
    }

    fn vectorization(&self) -> Option<std::num::NonZero<u8>> {
        None
    }
}

#[derive(new)]
pub struct Shape<Tensor: Expr, Dim: Expr>
where
    Tensor::Output: Strided,
    Dim::Output: Integer,
{
    pub tensor: Tensor,
    pub dim: Dim,
}

impl<Tensor: Expr, Dim: Expr> Expr for Shape<Tensor, Dim>
where
    Tensor::Output: Strided,
    Dim::Output: Integer,
{
    type Output = Dim::Output;

    fn expression_untyped(&self) -> super::Expression {
        Expression::Tensor(TensorExpression::Shape {
            tensor: Box::new(self.tensor.expression_untyped()),
            dim: Box::new(self.dim.expression_untyped()),
        })
    }

    fn vectorization(&self) -> Option<std::num::NonZero<u8>> {
        None
    }
}

#[derive(new)]
pub struct Length<Tensor: Expr, Out: Integer>
where
    Tensor::Output: Strided,
{
    pub tensor: Tensor,
    pub _out: PhantomData<Out>,
}

impl<Tensor: Expr, Out: Integer> Expr for Length<Tensor, Out>
where
    Tensor::Output: Strided,
{
    type Output = Out;

    fn expression_untyped(&self) -> super::Expression {
        Expression::Tensor(TensorExpression::Length {
            tensor: Box::new(self.tensor.expression_untyped()),
        })
    }

    fn vectorization(&self) -> Option<std::num::NonZero<u8>> {
        None
    }
}

#[derive(new)]
pub struct Rank<Tensor: Expr, Out: Integer>
where
    Tensor::Output: Strided,
{
    pub tensor: Tensor,
    pub _out: PhantomData<Out>,
}

impl<Tensor: Expr, Out: Integer> Expr for Rank<Tensor, Out>
where
    Tensor::Output: Strided,
{
    type Output = Out;

    fn expression_untyped(&self) -> super::Expression {
        Expression::Tensor(TensorExpression::Rank {
            tensor: Box::new(self.tensor.expression_untyped()),
        })
    }

    fn vectorization(&self) -> Option<std::num::NonZero<u8>> {
        None
    }
}

#[derive(new)]
pub struct IndexExpr<Tensor: Expr, Idx: Expr, Out: SquareType>
where
    Tensor::Output: Index<Idx::Output>,
    Idx::Output: Integer,
{
    pub tensor: Tensor,
    pub index: Idx,
    pub _out: PhantomData<Out>,
}

impl<Tensor: Expr, Idx: Expr, Out: SquareType> Expr for IndexExpr<Tensor, Idx, Out>
where
    Tensor::Output: Index<Idx::Output>,
    Idx::Output: Integer,
{
    type Output = Out;

    fn expression_untyped(&self) -> super::Expression {
        Expression::Tensor(TensorExpression::Index {
            tensor: Box::new(self.tensor.expression_untyped()),
            index: Box::new(self.index.expression_untyped()),
        })
    }

    fn vectorization(&self) -> Option<std::num::NonZero<u8>> {
        self.tensor.vectorization()
    }
}
