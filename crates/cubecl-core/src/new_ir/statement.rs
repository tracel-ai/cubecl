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
}

#[derive(Clone, Debug, PartialEq, new)]
pub struct Block<Ret: Expr>
where
    Ret::Output: SquareType,
{
    pub statements: Vec<Statement>,
    pub ret: Ret,
}

impl<Ret: Expr> Expr for Block<Ret>
where
    Ret::Output: SquareType,
{
    type Output = Ret::Output;

    fn expression_untyped(&self) -> Expression {
        Expression::Block {
            inner: self.statements.clone(),
            ret: Box::new(self.ret.expression_untyped()),
            vectorization: None,
            ty: <Ret::Output as SquareType>::ir_type(),
        }
    }

    fn vectorization(&self) -> Option<std::num::NonZero<u8>> {
        todo!()
    }
}
