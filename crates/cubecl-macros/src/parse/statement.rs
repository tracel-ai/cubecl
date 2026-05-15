use quote::{format_ident, quote};
use syn::{
    Expr, ExprArray, Ident, LitStr, Local, Macro, Pat, PatMacro, Stmt, Type, TypeReference,
    parse_quote, parse2,
};

use crate::{
    expression::Expression,
    parse::helpers::is_helper,
    scope::Context,
    statement::{DefineKind, Pattern, Statement},
};

impl Statement {
    pub fn from_stmt(stmt: Stmt, context: &mut Context) -> syn::Result<Self> {
        let statement = match stmt {
            Stmt::Local(mut local) => {
                if let Some((name, kind, init)) = parse_define_macro(&local) {
                    let init = Expression::from_expr(init, context)?;

                    Statement::Define {
                        name,
                        kind,
                        init: Box::new(init),
                    }
                } else {
                    let comptime_attr = local.attrs.iter().find(|it| is_helper(it));
                    // Syntax is weird without this
                    if let Some((init, attr)) = local.init.as_mut().zip(comptime_attr) {
                        match &mut *init.expr {
                            syn::Expr::Match(expr) => expr.attrs.push(attr.clone()),
                            syn::Expr::If(expr) => expr.attrs.push(attr.clone()),
                            _ => {}
                        }
                    }

                    let init = local
                        .init
                        .map(|init| Expression::from_expr(*init.expr, context))
                        .transpose()?
                        .map(Box::new);
                    let Pattern {
                        ident,
                        ty,
                        is_ref,
                        mutability,
                    } = parse_pat(local.pat)?;
                    let is_mut = mutability.is_some();
                    let is_const = init.as_ref().is_some_and(|init| init.is_const());

                    let variable =
                        context.push_variable(ident, ty, is_const && !is_mut, !is_ref && is_mut);
                    Self::Local { variable, init }
                }
            }
            Stmt::Expr(expr, semi) => {
                let expression = Box::new(Expression::from_expr(expr, context)?);
                Statement::Expression {
                    terminated: semi.is_some() || !expression.needs_terminator(),
                    expression,
                }
            }
            Stmt::Item(item) => Statement::Verbatim {
                tokens: quote![#item],
            },
            Stmt::Macro(val) => {
                let expression = parse_macros(val.mac, context)?;
                Statement::Expression {
                    expression: Box::new(expression),
                    terminated: val.semi_token.is_some(),
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
            mutability: ident.mutability,
        },
        Pat::Type(pat) => {
            let ty = *pat.ty;
            let is_ref = matches!(ty, Type::Reference(_));
            let mutability = match ty {
                Type::Reference(TypeReference { mutability, .. }) => mutability,
                _ => None,
            };
            let inner = parse_pat(*pat.pat)?;
            Pattern {
                ident: inner.ident,
                ty: Some(ty),
                is_ref: is_ref || inner.is_ref,
                mutability: mutability.or(inner.mutability),
            }
        }
        Pat::Wild(_) => Pattern {
            ident: format_ident!("_"),
            ty: None,
            is_ref: false,
            mutability: None,
        },
        pat => Err(syn::Error::new_spanned(
            pat.clone(),
            format!("Unsupported local pat: {pat:?}"),
        ))?,
    };
    Ok(res)
}

pub fn parse_define_macro(local: &Local) -> Option<(Ident, DefineKind, Expr)> {
    let Some(init) = &local.init else {
        return None;
    };
    let Pat::Macro(PatMacro { mac, .. }) = &local.pat else {
        return None;
    };
    let macro_ident = &mac.path.segments.last().unwrap().ident;
    let kind = if macro_ident == "size" {
        DefineKind::Size
    } else if macro_ident == "define" {
        DefineKind::Type
    } else {
        return None;
    };
    let name = parse2(mac.tokens.clone()).expect("Expected define macro to contain ident");
    Some((name, kind, *init.expr.clone()))
}

pub fn parse_macros(mac: Macro, context: &mut Context) -> syn::Result<Expression> {
    if mac.path.is_ident("comptime") {
        let tokens = &mac.tokens;
        Ok(Expression::Verbatim {
            tokens: quote![{#tokens}],
        })
    } else if [
        "panic",
        "assert",
        "assert_eq",
        "assert_ne",
        "todo",
        "unimplemented",
        "unreachable",
    ]
    .into_iter()
    .any(|target| mac.path.is_ident(&target))
    {
        Ok(Expression::RustMacro {
            ident: mac.path.segments.last().unwrap().ident.clone(),
            tokens: mac.tokens,
        })
    } else if mac.path.is_ident("debug_print") || mac.path.is_ident("seq") {
        let args = mac.tokens;
        let arg_exprs: ExprArray = parse_quote!([#args]);
        let args = arg_exprs
            .elems
            .into_iter()
            .map(|expr| Expression::from_expr(expr, context))
            .collect::<Result<_, _>>()?;
        Ok(Expression::ExpressionMacro {
            ident: mac.path.get_ident().cloned().unwrap(),
            args,
        })
    } else if mac.path.is_ident("comment") {
        let content = syn::parse2::<LitStr>(mac.tokens)?;
        Ok(Expression::Comment { content })
    } else if mac.path.is_ident("terminate") {
        Ok(Expression::Terminate)
    } else if mac.path.is_ident("intrinsic") {
        let closure: syn::ExprClosure = mac.parse_body()?;
        let arg = &closure.inputs[0];
        let block = *closure.body;
        let tokens = quote! {{
            let #arg = scope;
            #block
        }};

        Ok(Expression::Verbatim { tokens })
    } else {
        Err(syn::Error::new_spanned(
            mac,
            "Unsupported macro".to_string().as_str(),
        ))
    }
}
