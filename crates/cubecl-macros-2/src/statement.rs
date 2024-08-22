use proc_macro2::Span;
use quote::{format_ident, quote, quote_spanned, ToTokens};
use syn::{spanned::Spanned, Ident, Pat, Path, Stmt, Type};

use crate::{expression::Expression, ir_type, prefix_ir, scope::Context};

#[derive(Clone, Debug)]
pub enum Statement {
    Local {
        left: Box<Expression>,
        init: Option<Box<Expression>>,
        mutable: bool,
        ty: Option<Type>,
        span: Span,
    },
    Expression {
        expression: Box<Expression>,
        terminated: bool,
        span: Span,
    },
}

impl Statement {
    pub fn from_stmt(stmt: Stmt, context: &mut Context) -> syn::Result<Self> {
        let statement = match stmt {
            Stmt::Local(local) => {
                let span = local.span();
                let (ident, ty, mutable) = parse_pat(local.pat)?;
                let init = local
                    .init
                    .map(|init| Expression::from_expr(*init.expr, context))
                    .transpose()?
                    .map(Box::new);
                let is_const = init.as_ref().map(|init| init.is_const()).unwrap_or(false);
                let init_ty = init.as_ref().and_then(|init| init.ty());

                let variable = Box::new(Expression::Variable {
                    name: ident.clone(),
                    span: span.clone(),
                    ty: ty.clone(),
                });

                context.push_variable(ident, ty.clone(), is_const && !mutable);
                Self::Local {
                    left: variable,
                    init,
                    mutable,
                    ty,
                    span,
                }
            }
            Stmt::Expr(expr, semi) => Statement::Expression {
                terminated: semi.is_some(),
                span: expr.span(),
                expression: Box::new(Expression::from_expr(expr, context)?),
            },
            stmt => Err(syn::Error::new_spanned(stmt, "Unsupported statement"))?,
        };
        Ok(statement)
    }
}

pub fn parse_pat(pat: Pat) -> syn::Result<(Ident, Option<Type>, bool)> {
    let res = match pat {
        Pat::Ident(ident) => (ident.ident, None, ident.mutability.is_some()),
        Pat::Type(pat) => {
            let ty = *pat.ty;
            let (ident, _, mutable) = parse_pat(*pat.pat)?;
            (ident, Some(ty), mutable)
        }
        pat => Err(syn::Error::new_spanned(
            pat.clone(),
            format!("Unsupported local pat: {pat:?}"),
        ))?,
    };
    Ok(res)
}
