use super::{Expr, Expression, SquareType};
use core::fmt::Display;
use derive_more::derive::Display;
use std::ops::{Add, Deref, Mul};

#[derive(Clone, Copy, new, Display)]
pub struct Literal<T: Display + SquareType + Copy> {
    pub value: T,
}

impl<T: Display + SquareType + Clone + Copy> Expr for Literal<T> {
    type Output = T;

    fn expression_untyped(&self) -> Expression {
        Expression::Literal {
            value: self.value.to_string(),
            ty: <T as SquareType>::ir_type(),
        }
    }
}

impl<T: Mul<Output = T> + Display + SquareType + Clone + Copy> Mul<T> for Literal<T> {
    type Output = Literal<T>;

    fn mul(self, rhs: T) -> Self::Output {
        Literal {
            value: self.value * rhs,
        }
    }
}

impl<T: Add<Output = T> + Display + SquareType + Copy> Add<T> for Literal<T> {
    type Output = Literal<T>;

    fn add(self, rhs: T) -> Self::Output {
        Literal {
            value: self.value + rhs,
        }
    }
}

impl<T: Display + SquareType + Clone + Copy> Deref for Literal<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T: Display + SquareType + Clone + Copy> From<T> for Literal<T> {
    fn from(value: T) -> Self {
        Literal::new(value)
    }
}
