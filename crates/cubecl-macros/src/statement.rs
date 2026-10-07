use crate::{expression::Expression, scope::ManagedVar};
use proc_macro2::{Span, TokenStream};
use syn::{Ident, Type, token::Mut};

#[derive(Clone, Debug)]
pub enum Statement {
    Local {
        variable: ManagedVar,
        init: Option<Box<Expression>>,
        /// The source span of the statement, for its debug location.
        span: Option<Span>,
    },
    Define {
        name: Ident,
        kind: DefineKind,
        init: Box<Expression>,
    },
    Expression {
        expression: Box<Expression>,
        terminated: bool,
        /// The source span of the statement, for its debug location.
        span: Option<Span>,
    },
    Verbatim {
        tokens: TokenStream,
    },
}

impl Statement {
    /// The source span of the statement, if it has a debug location.
    pub fn span(&self) -> Option<Span> {
        match self {
            Statement::Local { span, .. } | Statement::Expression { span, .. } => *span,
            Statement::Define { .. } | Statement::Verbatim { .. } => None,
        }
    }
}

pub struct Pattern {
    pub ident: Ident,
    pub ty: Option<Type>,
    pub is_ref: bool,
    pub mutability: Option<Mut>,
}

#[derive(Clone, Copy, Debug)]
pub enum DefineKind {
    Type,
    Size,
}
