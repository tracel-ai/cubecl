use crate::prelude::*;
use std::{marker::PhantomData, ops::Index};

use super::{Container, Elem, Expr, Expression, RangeExpr, SquareType, Vectorization};

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
        vectorization: Vectorization,
    },
    Slice {
        ranges: Vec<SliceRange>,
        tensor: Box<Expression>,
    },
    __SliceRange(SliceRange),
}

#[derive(Clone, Debug, PartialEq)]
pub struct SliceRange {
    pub start: Box<Expression>,
    pub end: Option<Box<Expression>>,
    pub inclusive: bool,
}

impl TensorExpression {
    pub fn ir_type(&self) -> Elem {
        match self {
            TensorExpression::Stride { dim, .. } => dim.ir_type(),
            TensorExpression::Shape { dim, .. } => dim.ir_type(),
            TensorExpression::Length { .. } => Elem::UInt,
            TensorExpression::Rank { .. } => Elem::UInt,
            TensorExpression::Index { tensor, .. } => tensor.ir_type(),
            TensorExpression::Slice { tensor, .. } => tensor.ir_type(),
            TensorExpression::__SliceRange(SliceRange { start, .. }) => start.ir_type(),
        }
    }

    pub fn vectorization(&self) -> Vectorization {
        match self {
            TensorExpression::Stride { tensor, .. } => tensor.vectorization(),
            TensorExpression::Shape { tensor, .. } => tensor.vectorization(),
            TensorExpression::Length { tensor } => tensor.vectorization(),
            TensorExpression::Rank { tensor } => tensor.vectorization(),
            TensorExpression::Index { vectorization, .. } => *vectorization,
            TensorExpression::Slice { tensor, .. } => tensor.vectorization(),
            TensorExpression::__SliceRange(_) => None,
        }
    }
}

pub trait Strided {
    type Dims;
}

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
            vectorization: self.vectorization(),
        })
    }

    fn vectorization(&self) -> Option<std::num::NonZero<u8>> {
        self.tensor.vectorization()
    }
}

#[derive(new)]
pub struct SliceExpr<Start: Expr, Tensor: Expr>
where
    Tensor::Output: Strided,
    Start::Output: Integer,
{
    pub tensor: Tensor,
    pub ranges: Vec<Box<dyn Expr<Output = SliceRangeExpr<Start>>>>,
}

impl<Start: Expr, Tensor: Expr> Expr for SliceExpr<Start, Tensor>
where
    Tensor::Output: Strided + Container,
    Start::Output: Integer,
{
    type Output = Slice<Tensor, Start::Output>;

    fn expression_untyped(&self) -> Expression {
        let ranges = self
            .ranges
            .iter()
            .map(|range| {
                let range_expr = range.expression_untyped();
                match range_expr {
                    Expression::Tensor(TensorExpression::__SliceRange(range)) => range,
                    _ => panic!(),
                }
            })
            .collect();

        Expression::Tensor(TensorExpression::Slice {
            ranges,
            tensor: Box::new(self.tensor.expression_untyped()),
        })
    }

    fn vectorization(&self) -> Option<std::num::NonZero<u8>> {
        self.tensor.vectorization()
    }
}

#[derive(new)]
pub struct SliceRangeExpr<Start: Expr>
where
    Start::Output: Integer,
{
    pub start: Start,
    pub end: Option<Box<dyn Expr<Output = Start::Output>>>,
    pub inclusive: bool,
}

impl<Start: Expr> Expr for SliceRangeExpr<Start>
where
    Start::Output: Integer,
{
    type Output = Self;

    fn expression_untyped(&self) -> Expression {
        Expression::Tensor(TensorExpression::__SliceRange(SliceRange {
            start: Box::new(self.start.expression_untyped()),
            end: self
                .end
                .as_ref()
                .map(|it| it.expression_untyped())
                .map(Box::new),
            inclusive: self.inclusive,
        }))
    }

    fn vectorization(&self) -> Option<std::num::NonZero<u8>> {
        None
    }
}

impl<Start: Expr + 'static, End: Expr<Output = Start::Output> + 'static> From<RangeExpr<Start, End>>
    for SliceRangeExpr<Start>
where
    Start::Output: Integer,
{
    fn from(value: RangeExpr<Start, End>) -> Self {
        Self {
            start: value.start,
            end: Some(Box::new(value.end)),
            inclusive: value.inclusive,
        }
    }
}
