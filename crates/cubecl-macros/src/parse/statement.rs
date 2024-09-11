use quote::format_ident;
use syn::{Pat, Stmt, Type, TypeReference};

use crate::{
    expression::Expression,
    scope::Context,
    statement::{Pattern, Statement},
};

impl Statement {
    pub fn from_stmt(stmt: Stmt, context: &mut Context) -> syn::Result<Self> {
        let statement = match stmt {
            Stmt::Local(local) => {
                let init = local
                    .init
                    .map(|init| Expression::from_expr(*init.expr, context))
                    .transpose()?
                    .map(Box::new);
                let Pattern {
                    ident,
                    ty,
                    is_ref,
                    is_mut,
                } = parse_pat(local.pat)?;
                let is_const = init.as_ref().map(|init| init.is_const()).unwrap_or(false);

                let variable =
                    context.push_variable(ident, ty, is_const && !is_mut, is_ref, is_mut);
                Self::Local { variable, init }
            }
            Stmt::Expr(expr, semi) => {
                let expression = Box::new(Expression::from_expr(expr, context)?);
                Statement::Expression {
                    terminated: semi.is_some() || !expression.needs_terminator(),
                    expression,
                }
            }
            Stmt::Item(_) => Statement::Skip,
            stmt => Err(syn::Error::new_spanned(stmt, "Unsupported statement"))?,
        };
        Ok(statement)
    }
}

pub fn parse_pat(pat: Pat) -> syn::Result<Pattern> {
    let res = match pat {
        Pat::Ident(ident) => Pattern {
            ident: ident.ident,
            ty: None,
            is_ref: ident.by_ref.is_some(),
            is_mut: ident.mutability.is_some(),
        },
        Pat::Type(pat) => {
            let ty = *pat.ty;
            let is_ref = matches!(ty, Type::Reference(_));
            let ref_mut = matches!(
                ty,
                Type::Reference(TypeReference {
                    mutability: Some(_),
                    ..
                })
            );
            let inner = parse_pat(*pat.pat)?;
            Pattern {
                ident: inner.ident,
                ty: Some(ty),
                is_ref: is_ref || inner.is_ref,
                is_mut: ref_mut || inner.is_mut,
            }
        }
        Pat::Wild(_) => Pattern {
            ident: format_ident!("_"),
            ty: None,
            is_ref: false,
            is_mut: false,
        },
        pat => Err(syn::Error::new_spanned(
            pat.clone(),
            format!("Unsupported local pat: {pat:?}"),
        ))?,
    };
    Ok(res)
}
