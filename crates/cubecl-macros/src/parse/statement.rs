use quote::{format_ident, quote};
use syn::{
    Expr, ExprArray, Ident, LitStr, Local, Macro, Pat, PatMacro, Stmt, Type, TypeReference,
    parse_quote, parse2, spanned::Spanned,
};

use crate::{
    expression::Expression,
    parse::{asm::parse_asm_call, helpers::is_helper},
    scope::Context,
    statement::{DefineKind, Pattern, Statement},
};

impl Statement {
    pub fn from_stmt(stmt: Stmt, context: &mut Context) -> syn::Result<Self> {
        let span = Some(stmt.span());
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
                    Self::Local {
                        variable,
                        init,
                        span,
                    }
                }
            }
            Stmt::Expr(expr, semi) => {
                let expression = Box::new(Expression::from_expr(expr, context)?);
                Statement::Expression {
                    terminated: semi.is_some() || !expression.needs_terminator(),
                    expression,
                    span,
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
                    span,
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
        let warning = redundant_comptime_warning(&mac, context);
        Ok(Expression::Verbatim {
            tokens: quote![{#warning #tokens}],
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
        Ok(Expression::PanickingMacro {
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
        Ok(Expression::Terminate {
            closure_scope: context.current_closure_scope(),
            span: mac.span(),
        })
    } else if mac.path.is_ident("intrinsic") {
        context.is_intrinsic = true;
        let closure: syn::ExprClosure = mac.parse_body()?;
        let arg = &closure.inputs[0];
        let block = *closure.body;
        let tokens = quote! {{
            let #arg = scope;
            #block
        }};

        Ok(Expression::Verbatim { tokens })
    } else if mac.path.is_ident("gpu_asm") {
        Ok(Expression::Asm(parse_asm_call(context, mac.tokens)?))
    } else {
        let mut expand_name = mac.path.clone();
        let ident = &mut expand_name.segments.last_mut().unwrap().ident;
        *ident = format_ident!("__expand_{ident}");
        Ok(Expression::RawMacro {
            path: expand_name,
            args: mac.tokens,
        })
    }
}

/// A warning on a `comptime!` the macro doesn't need, when the `warn-redundant-comptime` feature is
/// on: its expression is already a comptime value, so the macro evaluates it on the host without
/// the wrapper and the kernel is the same.
///
/// It is a deprecation warning, the only warning a proc macro can raise on stable, spanned on the
/// `comptime!` so it points at the line to change.
#[cfg(feature = "warn-redundant-comptime")]
fn redundant_comptime_warning(mac: &Macro, context: &Context) -> proc_macro2::TokenStream {
    let Ok(expr) = parse2::<Expr>(mac.tokens.clone()) else {
        return quote![];
    };
    if context.in_short_circuit_operand || !is_redundant_comptime(&expr, context) {
        return quote![];
    }
    let span = mac.path.span();
    quote::quote_spanned! {span=>
        #[deprecated(
            note = "this `comptime!` isn't needed: its value is already comptime, so the macro evaluates it on the host without it"
        )]
        #[allow(non_upper_case_globals)]
        const redundant_comptime: bool = true;
        let _ = redundant_comptime;
    }
}

#[cfg(not(feature = "warn-redundant-comptime"))]
fn redundant_comptime_warning(_mac: &Macro, _context: &Context) -> proc_macro2::TokenStream {
    quote![]
}

/// Whether `comptime!(expr)` can go without changing the kernel. A lone literal or a lone comptime
/// local can't: where the macro reads a value in runtime code (a match arm, a branch's value) it
/// turns those into runtime variables, while it leaves a `comptime!` block and any other comptime
/// expression (a field, a method call, an operator) a host value.
#[cfg(feature = "warn-redundant-comptime")]
fn is_redundant_comptime(expr: &Expr, context: &Context) -> bool {
    match expr {
        Expr::Paren(paren) => is_redundant_comptime(&paren.expr, context),
        Expr::Group(group) => is_redundant_comptime(&group.expr, context),
        Expr::Lit(_) | Expr::Path(_) => false,
        expr => is_comptime_without_wrapper(expr, context),
    }
}

/// Whether the macro would hold `expr` as a comptime value without `comptime!`: a comptime local, a
/// field of one, a method call on one with comptime arguments, or an operator on comptime operands.
#[cfg(feature = "warn-redundant-comptime")]
fn is_comptime_without_wrapper(expr: &Expr, context: &Context) -> bool {
    use syn::{BinOp, UnOp};

    let operand =
        |expr: &Expr| matches!(expr, Expr::Lit(_)) || is_comptime_without_wrapper(expr, context);
    match expr {
        Expr::Path(path) if path.qself.is_none() => path
            .path
            .get_ident()
            .is_some_and(|name| context.is_const_variable(name)),
        Expr::Field(field) => is_comptime_without_wrapper(&field.base, context),
        Expr::Paren(paren) => is_comptime_without_wrapper(&paren.expr, context),
        Expr::Group(group) => is_comptime_without_wrapper(&group.expr, context),
        Expr::Reference(reference) => is_comptime_without_wrapper(&reference.expr, context),
        Expr::MethodCall(call) => {
            call.method != "runtime"
                && is_comptime_without_wrapper(&call.receiver, context)
                && call.args.iter().all(operand)
        }
        Expr::Unary(unary) => {
            matches!(unary.op, UnOp::Neg(_) | UnOp::Not(_)) && operand(&unary.expr)
        }
        Expr::Binary(binary) => {
            let assigns = matches!(
                binary.op,
                BinOp::AddAssign(_)
                    | BinOp::SubAssign(_)
                    | BinOp::MulAssign(_)
                    | BinOp::DivAssign(_)
                    | BinOp::RemAssign(_)
                    | BinOp::BitXorAssign(_)
                    | BinOp::BitAndAssign(_)
                    | BinOp::BitOrAssign(_)
                    | BinOp::ShlAssign(_)
                    | BinOp::ShrAssign(_)
            );
            !assigns && operand(&binary.left) && operand(&binary.right)
        }
        _ => false,
    }
}
