use std::num::NonZero;

use cubecl_core::{
    ir::Elem,
    new_ir::{Expression, SquareType, Statement},
};

#[allow(unused)]
pub fn var(name: &str, ty: Elem) -> Box<Expression> {
    Box::new(Expression::Variable {
        name: name.to_string(),
        ty,
        vectorization: None,
    })
}

#[allow(unused)]
pub fn vec_var(name: &str, ty: Elem, vectorization: u8) -> Expression {
    Expression::Variable {
        name: name.to_string(),
        ty,
        vectorization: NonZero::new(vectorization),
    }
}

#[allow(unused)]
pub fn lit<T: ToString + SquareType>(value: T) -> Expression {
    Expression::Literal {
        value: value.to_string(),
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
            left: Box::new(vec_var(name, right.ir_type(), vectorization)),
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
