use crate::{expression::Expression, scope::ManagedVar};
use proc_macro2::TokenStream;
use syn::{Ident, Type};

#[derive(Clone, Debug)]
pub enum Statement {
    Local {
        variable: ManagedVar,
        init: Option<Box<Expression>>,
    },
    Define {
        name: Ident,
        kind: DefineKind,
        init: Box<Expression>,
    },
    Expression {
        expression: Box<Expression>,
        terminated: bool,
    },
    Verbatim {
        tokens: TokenStream,
    },
}

pub struct Pattern {
    pub ident: Ident,
    pub ty: Option<Type>,
    pub is_ref: bool,
    pub is_mut: bool,
}

#[derive(Clone, Copy, Debug)]
pub enum DefineKind {
    Type,
    Size,
}
