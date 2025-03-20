use quote::format_ident;
use syn::{ExprArray, LitStr, Pat, Stmt, Type, TypeReference, parse_quote};

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
            Stmt::Macro(val) => {
                if val.mac.path.is_ident("comptime") {
                    Statement::Expression {
                        expression: Box::new(Expression::Verbatim {
                            tokens: val.mac.tokens,
                        }),
                        terminated: val.semi_token.is_some(),
                    }
                } else if val.mac.path.is_ident("debug_print") {
                    let args = val.mac.tokens;
                    let arg_exprs: ExprArray = parse_quote!([#args]);
                    let args = arg_exprs
                        .elems
                        .into_iter()
                        .map(|expr| Expression::from_expr(expr, context))
                        .collect::<Result<_, _>>()?;
                    Statement::Expression {
                        expression: Box::new(Expression::ExpressionMacro {
                            ident: val.mac.path.get_ident().cloned().unwrap(),
                            args,
                        }),
                        terminated: val.semi_token.is_some(),
                    }
                } else if val.mac.path.is_ident("comment") {
                    let content = syn::parse2::<LitStr>(val.mac.tokens)?;
                    Statement::Expression {
                        expression: Box::new(Expression::Comment { content }),
                        terminated: val.semi_token.is_some(),
                    }
                } else if val.mac.path.is_ident("terminate") {
                    Statement::Expression {
                        expression: Box::new(Expression::Terminate),
                        terminated: val.semi_token.is_some(),
                    }
                } else {
                    return Err(syn::Error::new_spanned(
                        val,
                        "Unsupported macro".to_string().as_str(),
                    ));
                }
            }
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
