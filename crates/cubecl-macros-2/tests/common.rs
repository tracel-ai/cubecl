use cubecl_core::{
    ir::Elem,
    new_ir::{Expression, SquareType, Statement},
};

#[allow(unused)]
pub fn var(name: &str, ty: Elem) -> Box<Expression> {
    Box::new(Expression::Variable {
        name: name.to_string(),
        ty,
    })
}

#[allow(unused)]
pub fn lit<T: ToString + SquareType>(value: T) -> Box<Expression> {
    Box::new(Expression::Literal {
        value: value.to_string(),
        ty: <T as SquareType>::ir_type(),
    })
}

#[allow(unused)]
pub fn local_init(
    name: &str,
    right: Box<Expression>,
    mutable: bool,
    ty: Option<Elem>,
) -> Statement {
    Statement::Local {
        variable: Box::new(Expression::Init {
            left: var(name, right.ir_type()),
            ty: right.ir_type(),
            right,
        }),
        mutable,
        ty,
    }
}

#[allow(unused)]
pub fn expr(expr: Box<Expression>) -> Statement {
    Statement::Expression(expr)
}
