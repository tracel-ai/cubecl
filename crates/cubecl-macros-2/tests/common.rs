use std::num::NonZero;

use cubecl_core::{
    ir::Elem,
    new_ir::{Block, Expr, Expression, Primitive, SquareType, Statement},
};

#[allow(unused)]
pub fn block(statements: Vec<Statement>, ret: Option<Expression>) -> Block {
    let ty = ret.as_ref().map(|ret| ret.ir_type()).unwrap_or(Elem::Unit);
    Block {
        inner: statements,
        ret: ret
            .map(Box::new)
            .unwrap_or_else(|| Box::new(().expression_untyped())),
        vectorization: None,
        ty,
    }
}

#[allow(unused)]
pub fn block_expr(statements: Vec<Statement>, ret: Option<Expression>) -> Expression {
    Expression::Block(block(statements, ret))
}

#[allow(unused)]
pub fn var(name: &str, ty: Elem) -> Box<Expression> {
    Box::new(Expression::Variable {
        name: name.to_string(),
        ty,
        vectorization: None,
    })
}

#[allow(unused)]
pub fn vec_var(name: &str, ty: Elem, vectorization: u8) -> Box<Expression> {
    Box::new(Expression::Variable {
        name: name.to_string(),
        ty,
        vectorization: NonZero::new(vectorization),
    })
}

#[allow(unused)]
pub fn lit<T: Primitive>(value: T) -> Expression {
    Expression::Literal {
        value: value.value(),
        ty: <T as SquareType>::ir_type(),
        vectorization: None,
    }
}

#[allow(unused)]
pub fn local_init(name: &str, right: Expression, mutable: bool, ty: Option<Elem>) -> Statement {
    Statement::Local {
        variable: Expression::Init {
            left: var(name, right.ir_type()),
            ty: right.ir_type(),
            right: Box::new(right),
            vectorization: None,
        },
        mutable,
        ty,
    }
}
#[allow(unused)]
pub fn init_vec(
    name: &str,
    right: Expression,
    mutable: bool,
    ty: Option<Elem>,
    vectorization: u8,
) -> Statement {
    Statement::Local {
        variable: Expression::Init {
            left: vec_var(name, right.ir_type(), vectorization),
            ty: right.ir_type(),
            right: Box::new(right),
            vectorization: NonZero::new(vectorization),
        },
        mutable,
        ty,
    }
}

#[allow(unused)]
pub fn expr(expr: Expression) -> Statement {
    Statement::Expression(expr)
}
