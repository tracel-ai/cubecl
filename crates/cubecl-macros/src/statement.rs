use crate::{expression::Expression, scope::ManagedVar};
use proc_macro2::Span;
use syn::{Ident, Type};

#[derive(Clone, Debug)]
pub enum Statement {
    Local {
        variable: ManagedVar,
        init: Option<Box<Expression>>,
    },
    /// Group of statements generated by desugaring
    Group {
        statements: Vec<Statement>,
    },
    Expression {
        expression: Box<Expression>,
        terminated: bool,
        span: Span,
    },
    Skip,
}

pub struct Pattern {
    pub ident: Ident,
    pub ty: Option<Type>,
    pub is_ref: bool,
    pub is_mut: bool,
}
