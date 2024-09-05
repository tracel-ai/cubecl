use core::{marker::PhantomData, ops::*};
use std::{
    num::NonZero,
    ops::{Shr, ShrAssign},
};

use super::{largest_common_vectorization, Expr, Expression, Operator, SquareType};

#[derive(new)]
pub struct BinaryOp<Left: Expr, Right: Expr, TOut>
where
    Left::Output: SquareType,
    Right::Output: SquareType,
{
    pub left: Left,
    pub right: Right,
    pub _out: PhantomData<TOut>,
}

#[derive(new)]
pub struct UnaryOp<In: Expr, TOut> {
    pub input: In,
    pub _out: PhantomData<TOut>,
}

macro_rules! bin_op {
    ($name:ident, $trait:ident, $operator:path) => {
        pub struct $name<Left: Expr, Right: Expr, TOut: SquareType>(
            pub BinaryOp<Left, Right, TOut>,
        )
        where
            Left::Output: $trait<Right::Output, Output = TOut> + SquareType,
            Right::Output: SquareType;

        impl<Left: Expr, Right: Expr, TOut: SquareType> $name<Left, Right, TOut>
        where
            Left::Output: $trait<Right::Output, Output = TOut> + SquareType,
            Right::Output: SquareType,
        {
            pub fn new(left: Left, right: Right) -> Self {
                Self(BinaryOp::new(left, right))
            }
        }

        impl<Left: Expr, Right: Expr, TOut: SquareType> Expr for $name<Left, Right, TOut>
        where
            Left::Output: $trait<Right::Output, Output = TOut> + SquareType,
            Right::Output: SquareType,
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
        cmp_op!($name, $trait, $operator, bool);
    };
    ($name:ident, $trait:ident, $operator:path, $out:path) => {
        pub struct $name<Left: Expr, Right: Expr>(pub BinaryOp<Left, Right, $out>)
        where
            Left::Output: $trait<Right::Output> + SquareType,
            Right::Output: SquareType;

        impl<Left: Expr, Right: Expr> $name<Left, Right>
        where
            Left::Output: $trait<Right::Output> + SquareType,
            Right::Output: SquareType,
        {
            pub fn new(left: Left, right: Right) -> Self {
                Self(BinaryOp::new(left, right))
            }
        }

        impl<Left: Expr, Right: Expr> Expr for $name<Left, Right>
        where
            Left::Output: $trait<Right::Output> + SquareType,
            Right::Output: SquareType,
        {
            type Output = $out;

            fn expression_untyped(&self) -> Expression {
                Expression::Binary {
                    left: Box::new(self.0.left.expression_untyped()),
                    right: Box::new(self.0.right.expression_untyped()),
                    operator: $operator,
                    ty: <Self::Output as SquareType>::ir_type(),
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
        pub struct $name<Left: Expr, Right: Expr>(pub BinaryOp<Left, Right, Left::Output>)
        where
            Left::Output: $trait<Right::Output> + SquareType,
            Right::Output: SquareType;

        impl<Left: Expr, Right: Expr> $name<Left, Right>
        where
            Left::Output: $trait<Right::Output> + SquareType,
            Right::Output: SquareType,
        {
            pub fn new(left: Left, right: Right) -> Self {
                Self(BinaryOp::new(left, right))
            }
        }

        impl<Left: Expr, Right: Expr> Expr for $name<Left, Right>
        where
            Left::Output: $trait<Right::Output> + SquareType,
            Right::Output: SquareType,
        {
            type Output = Left::Output;

            fn expression_untyped(&self) -> Expression {
                Expression::Binary {
                    left: Box::new(self.0.left.expression_untyped()),
                    right: Box::new(self.0.right.expression_untyped()),
                    operator: $operator,
                    ty: <Left::Output as SquareType>::ir_type(),
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
        pub struct $name<In: Expr, TOut>(pub UnaryOp<In, TOut>)
        where
            In::Output: $trait<$target = TOut> + SquareType;

        impl<In: Expr, TOut: SquareType> $name<In, TOut>
        where
            In::Output: $trait<$target = TOut> + SquareType,
        {
            pub fn new(input: In) -> Self {
                Self(UnaryOp::new(input))
            }
        }

        impl<In: Expr, TOut: SquareType> Expr for $name<In, TOut>
        where
            In::Output: $trait<$target = TOut> + SquareType,
        {
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

cmp_op!(MinExpr, PartialOrd, Operator::Min, Left::Output);
cmp_op!(MaxExpr, PartialOrd, Operator::Max, Left::Output);

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

pub struct DerefExpr<In: Expr, TOut>(pub UnaryOp<In, TOut>)
where
    In::Output: SquareType;

impl<In: Expr, TOut: SquareType> DerefExpr<In, TOut>
where
    In::Output: SquareType,
{
    pub fn new(input: In) -> Self {
        Self(UnaryOp::new(input))
    }
}

impl<In: Expr, TOut: SquareType> Expr for DerefExpr<In, TOut>
where
    In::Output: SquareType,
{
    type Output = TOut;

    fn expression_untyped(&self) -> Expression {
        let in_ty = In::Output::ir_type();
        let out_ty = TOut::ir_type();
        if in_ty != out_ty {
            Expression::Cast {
                from: Box::new(self.0.input.expression_untyped()),
                vectorization: self.vectorization(),
                to: TOut::ir_type(),
            }
        } else {
            self.0.input.expression_untyped()
        }
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        self.0.input.vectorization()
    }
}

pub struct AndExpr<Left: Expr<Output = bool>, Right: Expr<Output = bool>>(
    pub BinaryOp<Left, Right, bool>,
);
pub struct OrExpr<Left: Expr<Output = bool>, Right: Expr<Output = bool>>(
    pub BinaryOp<Left, Right, bool>,
);

impl<Left: Expr<Output = bool>, Right: Expr<Output = bool>> AndExpr<Left, Right> {
    pub fn new(left: Left, right: Right) -> Self {
        Self(BinaryOp::new(left, right))
    }
}

impl<Left: Expr<Output = bool>, Right: Expr<Output = bool>> OrExpr<Left, Right> {
    pub fn new(left: Left, right: Right) -> Self {
        Self(BinaryOp::new(left, right))
    }
}

impl<Left: Expr<Output = bool>, Right: Expr<Output = bool>> Expr for AndExpr<Left, Right> {
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

impl<Left: Expr<Output = bool>, Right: Expr<Output = bool>> Expr for OrExpr<Left, Right> {
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

pub mod new {
    use std::ops::{Add, Div, Mul, Rem, Sub};

    use cubecl_common::operator::Operator;

    use crate::{
        new_ir::{backend::Backend, largest_common_vectorization, CubeType, NewExpr, SquareType},
        prelude::ExpandElement,
    };
    use cubecl_macros::expression;

    macro_rules! bin_op {
        ($name:ident, $trait:ident, $op:ident) => {
            #[expression(output = <Left::Output as $trait<Right::Output>>::Output)]
            pub fn $name<Left: NewExpr<B>, Right: NewExpr<B>, B: Backend>(
                left: &Left,
                right: &Right,
                backend: &mut B,
            ) -> ExpandElement
            where
                Left::Output: $trait<Right::Output>,
                <Left::Output as $trait<Right::Output>>::Output: CubeType + SquareType,
            {
                backend.expand_binop(
                    left,
                    right,
                    Operator::$op,
                    <Left::Output as $trait<Right::Output>>::Output::ir_type(),
                    largest_common_vectorization(left.vectorization(), right.vectorization()),
                )
            }
        };
    }

    bin_op!(add_expr, Add, Add);
    bin_op!(sub_expr, Sub, Sub);
    bin_op!(mul_expr, Mul, Mul);
    bin_op!(div_expr, Div, Div);
    bin_op!(rem_expr, Rem, Rem);
}
