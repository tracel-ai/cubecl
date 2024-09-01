use std::marker::PhantomData;

use crate::prelude::*;

use super::{Expr, Expression, SquareType, Vectorization};

#[derive(new)]
pub struct ArrayInit<T: SquareType> {
    pub size: u32,
    pub vectorization: Vectorization,
    pub _type: PhantomData<T>,
}

impl<T: SquareType> Expr for ArrayInit<T> {
    type Output = Array<T>;

    fn expression_untyped(&self) -> super::Expression {
        Expression::ArrayInit {
            size: self.size,
            ty: T::ir_type(),
            vectorization: self.vectorization(),
        }
    }

    fn vectorization(&self) -> Option<std::num::NonZero<u8>> {
        self.vectorization
    }
}
