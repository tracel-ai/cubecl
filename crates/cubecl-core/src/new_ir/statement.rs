use std::marker::PhantomData;

use crate::ir::Elem;

use super::Expression;

#[derive(Clone, Debug, PartialEq)]
pub enum Statement {
    Local {
        variable: Box<Expression>,
        mutable: bool,
        ty: Option<Elem>,
    },
    Expression(Box<Expression>),
    Return(Box<Expression>),
}

#[derive(Clone, Debug, PartialEq, new)]
pub struct Block<T> {
    pub statements: Vec<Statement>,
    pub _ty: PhantomData<T>,
}
