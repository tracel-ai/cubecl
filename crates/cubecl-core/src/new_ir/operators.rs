use core::{marker::PhantomData, ops::*};
use std::{
    num::NonZero,
    ops::{Shr, ShrAssign},
};

use super::{largest_common_vectorization, Expr, Expression, Operator, SquareType};

#[derive(new)]
pub struct BinaryOp<TLeft, TRight, TOut> {
    pub left: Box<dyn Expr<Output = TLeft>>,
    pub right: Box<dyn Expr<Output = TRight>>,
    pub _out: PhantomData<TOut>,
}

#[derive(new)]
pub struct UnaryOp<TIn, TOut> {
    pub input: Box<dyn Expr<Output = TIn>>,
    pub _out: PhantomData<TOut>,
}

macro_rules! bin_op {
    ($name:ident, $trait:ident, $operator:path) => {
        pub struct $name<TLeft: SquareType, TRight: SquareType, TOut: SquareType>(
            pub BinaryOp<TLeft, TRight, TOut>,
        )
        where
            TLeft: $trait<TRight, Output = TOut>;

        impl<TLeft: SquareType, TRight: SquareType, TOut: SquareType> Expr
            for $name<TLeft, TRight, TOut>
        where
            TLeft: $trait<TRight, Output = TOut>,
        {
            type Output = TOut;

            fn expression_untyped(&self) -> Expression {
                Expression::Binary {
                    left: Box::new(self.0.left.expression_untyped()),
                    right: Box::new(self.0.right.expression_untyped()),
                    operator: $operator,
                    ty: <TOut as SquareType>::ir_type(),
                    vectorization: self.vectorization(),
                }
            }

            fn vectorization(&self) -> Option<NonZero<u8>> {
                largest_common_vectorization(
                    self.0.left.vectorization(),
                    self.0.right.vectorization(),
                )
            }
        }
    };
}

macro_rules! cmp_op {
    ($name:ident, $trait:ident, $operator:path) => {
        pub struct $name<TLeft: $trait<TRight>, TRight>(pub BinaryOp<TLeft, TRight, bool>);

        impl<TLeft: $trait<TRight>, TRight> Expr for $name<TLeft, TRight> {
            type Output = bool;

            fn expression_untyped(&self) -> Expression {
                Expression::Binary {
                    left: Box::new(self.0.left.expression_untyped()),
                    right: Box::new(self.0.right.expression_untyped()),
                    operator: $operator,
                    ty: <bool as SquareType>::ir_type(),
                    vectorization: self.vectorization(),
                }
            }

            fn vectorization(&self) -> Option<NonZero<u8>> {
                largest_common_vectorization(
                    self.0.left.vectorization(),
                    self.0.right.vectorization(),
                )
            }
        }
    };
}

macro_rules! assign_bin_op {
    ($name:ident, $trait:ident, $operator:path) => {
        pub struct $name<TLeft, TRight>(pub BinaryOp<TLeft, TRight, TLeft>)
        where
            TLeft: $trait<TRight> + SquareType;

        impl<TLeft: $trait<TRight> + SquareType, TRight> Expr for $name<TLeft, TRight> {
            type Output = TLeft;

            fn expression_untyped(&self) -> Expression {
                Expression::Binary {
                    left: Box::new(self.0.left.expression_untyped()),
                    right: Box::new(self.0.right.expression_untyped()),
                    operator: $operator,
                    ty: <TLeft as SquareType>::ir_type(),
                    vectorization: self.vectorization(),
                }
            }

            fn vectorization(&self) -> Option<NonZero<u8>> {
                largest_common_vectorization(
                    self.0.left.vectorization(),
                    self.0.right.vectorization(),
                )
            }
        }
    };
}

macro_rules! unary_op {
    ($name:ident, $trait:ident, $operator:path, $target:ident) => {
        pub struct $name<TIn: $trait<$target = TOut>, TOut>(pub UnaryOp<TIn, TOut>);

        impl<TIn: $trait<$target = TOut>, TOut: SquareType> Expr for $name<TIn, TOut> {
            type Output = TOut;

            fn expression_untyped(&self) -> Expression {
                Expression::Unary {
                    input: Box::new(self.0.input.expression_untyped()),
                    operator: $operator,
                    ty: <TOut as SquareType>::ir_type(),
                    vectorization: self.vectorization(),
                }
            }

            fn vectorization(&self) -> Option<NonZero<u8>> {
                self.0.input.vectorization()
            }
        }
    };
}

// Arithmetic
bin_op!(AddExpr, Add, Operator::Add);
bin_op!(SubExpr, Sub, Operator::Sub);
bin_op!(MulExpr, Mul, Operator::Mul);
bin_op!(DivExpr, Div, Operator::Div);
bin_op!(RemExpr, Rem, Operator::Rem);

// Comparison
cmp_op!(EqExpr, PartialEq, Operator::Eq);
cmp_op!(NeExpr, PartialEq, Operator::Ne);
cmp_op!(LtExpr, PartialOrd, Operator::Lt);
cmp_op!(LeExpr, PartialOrd, Operator::Le);
cmp_op!(GeExpr, PartialOrd, Operator::Ge);
cmp_op!(GtExpr, PartialOrd, Operator::Gt);

// Boolean
bin_op!(BitXorExpr, BitXor, Operator::BitXor);
bin_op!(BitAndExpr, BitAnd, Operator::BitAnd);
bin_op!(BitOrExpr, BitOr, Operator::BitOr);

// Shift
bin_op!(ShlExpr, Shl, Operator::Shl);
bin_op!(ShrExpr, Shr, Operator::Shr);

// Arithmetic assign
assign_bin_op!(AddAssignExpr, AddAssign, Operator::AddAssign);
assign_bin_op!(SubAssignExpr, SubAssign, Operator::SubAssign);
assign_bin_op!(MulAssignExpr, MulAssign, Operator::MulAssign);
assign_bin_op!(DivAssignExpr, DivAssign, Operator::DivAssign);
assign_bin_op!(RemAssignExpr, RemAssign, Operator::RemAssign);

// Boolean assign
assign_bin_op!(BitXorAssignExpr, BitXorAssign, Operator::BitXorAssign);
assign_bin_op!(BitAndAssignExpr, BitAndAssign, Operator::BitAndAssign);
assign_bin_op!(BitOrAssignExpr, BitOrAssign, Operator::BitOrAssign);

// Shift assign
assign_bin_op!(ShlAssignExpr, ShlAssign, Operator::ShlAssign);
assign_bin_op!(ShrAssignExpr, ShrAssign, Operator::ShrAssign);

unary_op!(NotExpr, Not, Operator::Not, Output);
unary_op!(NegExpr, Neg, Operator::Neg, Output);
unary_op!(DerefExpr, Deref, Operator::Deref, Target);

pub struct AndExpr(pub BinaryOp<bool, bool, bool>);
pub struct OrExpr(pub BinaryOp<bool, bool, bool>);

impl Expr for AndExpr {
    type Output = bool;

    fn expression_untyped(&self) -> Expression {
        Expression::Binary {
            left: Box::new(self.0.left.expression_untyped()),
            operator: Operator::And,
            right: Box::new(self.0.right.expression_untyped()),
            ty: bool::ir_type(),
            vectorization: self.vectorization(),
        }
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        None
    }
}

impl Expr for OrExpr {
    type Output = bool;

    fn expression_untyped(&self) -> Expression {
        Expression::Binary {
            left: Box::new(self.0.left.expression_untyped()),
            operator: Operator::Or,
            right: Box::new(self.0.right.expression_untyped()),
            ty: bool::ir_type(),
            vectorization: self.vectorization(),
        }
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        None
    }
}
