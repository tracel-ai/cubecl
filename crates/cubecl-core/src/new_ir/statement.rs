use std::num::NonZero;

use crate::ir::Elem;

use super::{Block, Expr, Expression, SquareType};

#[derive(Clone, Debug, PartialEq)]
pub enum Statement {
    Local {
        variable: Expression,
        mutable: bool,
        ty: Option<Elem>,
    },
    Expression(Expression),
}

impl Statement {
    pub fn deep_clone(&self) -> Statement {
        match self {
            Statement::Local {
                variable,
                mutable,
                ty,
            } => Statement::Local {
                variable: variable.deep_clone(),
                mutable: *mutable,
                ty: *ty,
            },
            Statement::Expression(expr) => Statement::Expression(expr.deep_clone()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, new)]
pub struct BlockExpr<Ret: Expr>
where
    Ret::Output: SquareType,
{
    pub statements: Vec<Statement>,
    pub ret: Ret,
}

impl<Ret: Expr> Expr for BlockExpr<Ret>
where
    Ret::Output: SquareType,
{
    type Output = Ret::Output;

    fn expression_untyped(&self) -> Expression {
        Expression::Block(Block {
            inner: self.statements.clone(),
            ret: Box::new(self.ret.expression_untyped()),
            vectorization: None,
            ty: <Ret::Output as SquareType>::ir_type(),
        })
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        self.ret.vectorization()
    }
}
