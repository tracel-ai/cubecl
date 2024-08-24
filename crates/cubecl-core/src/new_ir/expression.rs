use crate::ir::Elem;
use std::{marker::PhantomData, num::NonZero};

use super::{largest_common_vectorization, Operator, SquareType, Statement, TypeEq};

type Vectorization = Option<NonZero<u8>>;

#[derive(Clone, Debug, PartialEq)]
pub enum Expression {
    Binary {
        left: Box<Expression>,
        operator: Operator,
        right: Box<Expression>,
        vectorization: Vectorization,
        ty: Elem,
    },
    Unary {
        input: Box<Expression>,
        operator: Operator,
        vectorization: Vectorization,
        ty: Elem,
    },
    Variable {
        name: String,
        vectorization: Vectorization,
        ty: Elem,
    },
    FieldAccess {
        base: Box<Expression>,
        name: String,
        vectorization: Vectorization,
        ty: Elem,
    },
    Literal {
        // Stringified value for outputting directly to generated code
        value: String,
        vectorization: Vectorization,
        ty: Elem,
    },
    Assigment {
        left: Box<Expression>,
        right: Box<Expression>,
        vectorization: Vectorization,
        ty: Elem,
    },
    /// Local variable initializer
    Init {
        left: Box<Expression>,
        right: Box<Expression>,
        vectorization: Vectorization,
        ty: Elem,
    },
    Block {
        inner: Vec<Statement>,
        ret: Option<Box<Expression>>,
        vectorization: Vectorization,
        ty: Elem,
    },
    Break,
    Cast {
        from: Box<Expression>,
        vectorization: Vectorization,
        to: Elem,
    },
    Continue,
    ForLoop {
        from: Box<Expression>,
        to: Box<Expression>,
        step: Box<Expression>,
        unroll: bool,
        variable: Box<Expression>,
        block: Vec<Statement>,
    },
}

impl Expression {
    pub fn ir_type(&self) -> Elem {
        match self {
            Expression::Binary { ty, .. } => *ty,
            Expression::Unary { ty, .. } => *ty,
            Expression::Variable { ty, .. } => *ty,
            Expression::Literal { ty, .. } => *ty,
            Expression::Assigment { ty, .. } => *ty,
            Expression::Init { ty, .. } => *ty,
            Expression::Block { ret, .. } => {
                ret.as_ref().map(|ret| ret.ir_type()).unwrap_or(Elem::UInt)
            }
            Expression::Cast { to, .. } => *to,
            Expression::Break | Expression::Continue | Expression::ForLoop { .. } => Elem::UInt,
            Expression::FieldAccess { ty, .. } => *ty,
        }
    }
}

pub trait Expr {
    type Output;

    fn expression_untyped(&self) -> Expression;
    fn vectorization(&self) -> Option<NonZero<u8>>;
}

#[derive(Debug, new, Hash)]
pub struct Variable<T: SquareType> {
    pub name: &'static str,
    pub vectorization: Option<NonZero<u8>>,
    pub _type: PhantomData<T>,
}

impl<T: SquareType> Copy for Variable<T> {}
#[allow(clippy::non_canonical_clone_impl)]
impl<T: SquareType> Clone for Variable<T> {
    fn clone(&self) -> Self {
        Self {
            name: self.name,
            vectorization: self.vectorization,
            _type: PhantomData,
        }
    }
}

impl<T: SquareType> Expr for Variable<T> {
    type Output = T;

    fn expression_untyped(&self) -> Expression {
        Expression::Variable {
            name: self.name.to_string(),
            ty: <T as SquareType>::ir_type(),
            vectorization: self.vectorization(),
        }
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        self.vectorization
    }
}

#[derive(new, Hash)]
pub struct FieldAccess<T: SquareType, TBase: Expr> {
    pub base: TBase,
    pub name: &'static str,
    pub _type: PhantomData<T>,
}

impl<T: SquareType, TBase: Expr + Clone> Clone for FieldAccess<T, TBase> {
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            name: self.name,
            _type: PhantomData,
        }
    }
}

impl<T: SquareType, TBase: Expr> Expr for FieldAccess<T, TBase> {
    type Output = T;

    fn expression_untyped(&self) -> Expression {
        Expression::FieldAccess {
            base: Box::new(self.base.expression_untyped()),
            name: self.name.to_string(),
            ty: <T as SquareType>::ir_type(),
            vectorization: self.vectorization(),
        }
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        self.base.vectorization()
    }
}

pub struct Assignment<Left: Expr, Right: Expr>
where
    Left::Output: SquareType + TypeEq<Right::Output>,
{
    pub left: Left,
    pub right: Right,
}

impl<Left: Expr, Right: Expr> Expr for Assignment<Left, Right>
where
    Left::Output: SquareType + TypeEq<Right::Output>,
{
    type Output = ();

    fn expression_untyped(&self) -> Expression {
        Expression::Assigment {
            left: Box::new(self.left.expression_untyped()),
            right: Box::new(self.right.expression_untyped()),
            ty: <Left::Output as SquareType>::ir_type(),
            vectorization: self.vectorization(),
        }
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        largest_common_vectorization(self.left.vectorization(), self.right.vectorization())
    }
}

pub struct Initializer<Left: Expr, Right: Expr>
where
    Right::Output: SquareType + TypeEq<Left::Output>,
{
    pub left: Left,
    pub right: Right,
}

impl<Left: Expr, Right: Expr> Expr for Initializer<Left, Right>
where
    Right::Output: SquareType + TypeEq<Left::Output>,
{
    type Output = Right::Output;

    fn expression_untyped(&self) -> Expression {
        Expression::Init {
            left: Box::new(self.left.expression_untyped()),
            right: Box::new(self.right.expression_untyped()),
            ty: <Right::Output as SquareType>::ir_type(),
            vectorization: self.vectorization(),
        }
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        self.right.vectorization()
    }
}

pub struct Cast<From: Expr, TTo: SquareType>
where
    From::Output: SquareType,
{
    pub from: From,
    pub _to: PhantomData<TTo>,
}

impl<From: Expr, TTo: SquareType> Expr for Cast<From, TTo>
where
    From::Output: SquareType,
{
    type Output = TTo;

    fn expression_untyped(&self) -> Expression {
        Expression::Cast {
            from: Box::new(self.from.expression_untyped()),
            to: <TTo as SquareType>::ir_type(),
            vectorization: self.vectorization(),
        }
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        self.from.vectorization()
    }
}

impl<T: Expr> Expr for Box<T> {
    type Output = T::Output;

    fn expression_untyped(&self) -> Expression {
        let this: &T = self;
        this.expression_untyped()
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        let this: &T = self;
        this.vectorization()
    }
}
