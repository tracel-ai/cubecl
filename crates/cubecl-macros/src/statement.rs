use crate::{expression::Expression, scope::ManagedVar};
use syn::{Ident, Type};

#[derive(Clone, Debug)]
pub enum Statement {
    Local {
        variable: ManagedVar,
        init: Option<Box<Expression>>,
    },
    Expression {
        expression: Box<Expression>,
        terminated: bool,
    },
    Skip,
}

pub struct Pattern {
    pub ident: Ident,
    pub ty: Option<Type>,
    pub is_ref: bool,
    pub is_mut: bool,
}
