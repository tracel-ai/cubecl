use std::num::NonZero;

use cubecl_common::operator::Operator;
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote, quote_spanned, ToTokens};
use syn::{parse::Parse, spanned::Spanned, Expr, Ident, Lit, Member, Pat, Path, Type};

use crate::{
    ir_type, prefix_ir,
    scope::{Context, ManagedVar},
    statement::{parse_pat, Statement},
};

#[derive(Clone, Debug)]
pub enum Expression {
    Binary {
        left: Box<Expression>,
        operator: Operator,
        right: Box<Expression>,
        ty: Option<Type>,
        span: Span,
    },
    Unary {
        input: Box<Expression>,
        operator: Operator,
        ty: Option<Type>,
        span: Span,
    },
    Variable {
        name: Ident,
        ty: Option<Type>,
        span: Span,
    },
    ConstVariable {
        name: Ident,
        ty: Option<Type>,
        span: Span,
    },
    FieldAccess {
        base: Box<Expression>,
        field: Member,
        struct_ty: Type,
        span: Span,
    },
    Literal {
        value: Lit,
        ty: Type,
        span: Span,
    },
    Assigment {
        left: Box<Expression>,
        right: Box<Expression>,
        ty: Option<Type>,
        span: Span,
    },
    Init {
        left: Box<Expression>,
        right: Box<Expression>,
        ty: Option<Type>,
        span: Span,
    },
    Block {
        inner: Vec<Statement>,
        ret: Option<Box<Expression>>,
        ty: Option<Type>,
        span: Span,
    },
    FunctionCall {
        func: Box<Expression>,
        args: Vec<Expression>,
        span: Span,
    },
    Cast {
        from: Box<Expression>,
        to: Type,
        span: Span,
    },
    Break {
        span: Span,
    },
    /// Tokens not relevant to parsing
    Verbatim {
        tokens: TokenStream,
    },
    Continue {
        span: Span,
    },
    ForLoop {
        from: Box<Expression>,
        to: Box<Expression>,
        step: Option<Box<Expression>>,
        unroll: Box<Expression>,
        var_name: syn::Ident,
        var_ty: Option<syn::Type>,
        var_mut: bool,
        block: Vec<Statement>,
        span: Span,
    },
}

impl Expression {
    pub fn ty(&self) -> Option<Type> {
        match self {
            Expression::Binary { ty, .. } => ty.clone(),
            Expression::Unary { ty, .. } => ty.clone(),
            Expression::Variable { ty, .. } => ty.clone(),
            Expression::ConstVariable { ty, .. } => ty.clone(),
            Expression::Literal { ty, .. } => Some(ty.clone()),
            Expression::Assigment { ty, .. } => ty.clone(),
            Expression::Verbatim { .. } => None,
            Expression::Init { ty, .. } => ty.clone(),
            Expression::Block { ty, .. } => ty.clone(),
            Expression::FunctionCall { .. } => None,
            Expression::Break { .. } => None,
            Expression::Cast { to, .. } => Some(to.clone()),
            Expression::Continue { .. } => None,
            Expression::ForLoop { .. } => None,
            Expression::FieldAccess { .. } => None,
        }
    }

    pub fn is_const(&self) -> bool {
        match self {
            Expression::Literal { .. } => true,
            Expression::Verbatim { .. } => true,
            Expression::ConstVariable { .. } => true,
            Expression::FieldAccess { base, .. } => base.is_const(),
            _ => false,
        }
    }

    pub fn as_const(&self) -> Option<TokenStream> {
        match self {
            Expression::Literal { value, .. } => Some(quote![#value]),
            Expression::Verbatim { tokens, .. } => Some(tokens.clone()),
            Expression::ConstVariable { name, .. } => Some(quote![#name]),
            Expression::FieldAccess { base, field, .. } => {
                base.as_const().map(|base| quote![#base.#field])
            }
            _ => None,
        }
    }
}
