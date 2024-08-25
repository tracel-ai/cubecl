use std::marker::PhantomData;

use crate::ir::Elem;

use super::{Expr, Expression, SquareType};

#[derive(Clone, Debug, PartialEq)]
pub enum Statement {
    Local {
        variable: Expression,
        mutable: bool,
        ty: Option<Elem>,
    },
    Expression(Expression),
    Return(Expression),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Block<T: SquareType> {
    pub statements: Vec<Statement>,
    pub ret: Option<Expression>,
    pub _ty: PhantomData<T>,
}

impl<T: SquareType> Block<T> {
    pub fn new(mut statements: Vec<Statement>) -> Self {
        let ret = match statements.pop() {
            Some(Statement::Return(ret)) => Some(ret),
            Some(last) => {
                statements.push(last);
                None
            }
            _ => None,
        };
        Self {
            statements,
            ret,
            _ty: PhantomData,
        }
    }
}

impl<T: SquareType> Expr for Block<T> {
    type Output = T;

    fn expression_untyped(&self) -> Expression {
        Expression::Block {
            inner: self.statements.clone(),
            ret: self.ret.as_ref().map(ToOwned::to_owned).map(Box::new),
            vectorization: None,
            ty: <T as SquareType>::ir_type(),
        }
    }

    fn vectorization(&self) -> Option<std::num::NonZero<u8>> {
        todo!()
    }
}
