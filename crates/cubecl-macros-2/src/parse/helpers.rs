use darling::FromMeta;
use syn::{parse_quote, visit_mut::VisitMut, Attribute, Expr};

use crate::{expression::Expression, scope::Context};

pub struct Unroll {
    pub value: Expression,
}

impl Unroll {
    pub fn from_attributes(
        attrs: &[Attribute],
        context: &mut Context,
    ) -> Option<syn::Result<Self>> {
        #[derive(FromMeta)]
        struct NameVal {
            pub value: Expr,
        }

        let attr = attrs.iter().find(|attr| attr.path().is_ident("unroll"))?;
        let res = match &attr.meta {
            syn::Meta::Path(_) => Self {
                value: Expression::from_expr(parse_quote![true], context).unwrap(),
            },
            syn::Meta::List(list) => {
                let expr = syn::parse2(list.tokens.clone())
                    .and_then(|expr| Expression::from_expr(expr, context));
                let expr = match expr {
                    Ok(expr) => expr,
                    Err(e) => return Some(Err(e)),
                };
                Self { value: expr }
            }
            meta => {
                let expr = NameVal::from_meta(meta)
                    .map_err(Into::into)
                    .and_then(|expr| Expression::from_expr(expr.value, context));
                let expr = match expr {
                    Ok(expr) => expr,
                    Err(e) => return Some(Err(e)),
                };
                Self { value: expr }
            }
        };
        Some(Ok(res))
    }
}

pub struct RemoveHelpers;

impl VisitMut for RemoveHelpers {
    fn visit_fn_arg_mut(&mut self, i: &mut syn::FnArg) {
        match i {
            syn::FnArg::Receiver(recv) => recv.attrs.retain(|it| !is_comptime_attr(it)),
            syn::FnArg::Typed(typed) => typed.attrs.retain(|it| !is_comptime_attr(it)),
        }
    }

    fn visit_expr_for_loop_mut(&mut self, i: &mut syn::ExprForLoop) {
        i.attrs.retain(|attr| !is_unroll_attr(attr))
    }
}

pub fn is_comptime_attr(attr: &Attribute) -> bool {
    attr.path().is_ident("comptime")
}

pub fn is_unroll_attr(attr: &Attribute) -> bool {
    attr.path().is_ident("unroll")
}

pub fn is_helper(attr: &Attribute) -> bool {
    is_comptime_attr(attr) || is_unroll_attr(attr)
}
