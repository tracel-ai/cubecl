use darling::FromMeta;
use syn::{
    Attribute, Expr, Stmt, parse_quote,
    visit_mut::{self, VisitMut},
};

use crate::{
    expression::Expression, parse::statement::parse_define_macro, paths::prelude_type,
    scope::Context, statement::DefineKind,
};

pub struct Unroll {
    pub value: Expression,
    pub always_true: bool,
}

impl Unroll {
    pub fn from_attributes(
        attrs: &[Attribute],
        context: &mut Context,
    ) -> syn::Result<Option<Self>> {
        #[derive(FromMeta)]
        struct NameVal {
            pub value: Expr,
        }

        let attr = attrs.iter().find(|attr| attr.path().is_ident("unroll"));
        let attr = match attr {
            Some(attr) => attr,
            None => return Ok(None),
        };

        let res = match &attr.meta {
            syn::Meta::Path(_) => Self {
                value: Expression::from_expr(parse_quote![true], context).unwrap(),
                always_true: true,
            },
            syn::Meta::List(list) => {
                let expr = syn::parse2(list.tokens.clone())?;
                let expr = Expression::from_expr(expr, context)?;
                Self {
                    value: expr,
                    always_true: false,
                }
            }
            meta => {
                let expr = NameVal::from_meta(meta)?;
                let expr = Expression::from_expr(expr.value, context)?;
                Self {
                    value: expr,
                    always_true: false,
                }
            }
        };
        Ok(Some(res))
    }

    pub fn unroll_expr(attrs: &[Attribute]) -> Option<Expr> {
        #[derive(FromMeta)]
        struct NameVal {
            pub value: Expr,
        }

        let attr = attrs.iter().find(|attr| attr.path().is_ident("unroll"))?;

        match &attr.meta {
            syn::Meta::Path(_) => None,
            syn::Meta::List(list) => syn::parse2(list.tokens.clone()).ok(),
            meta => Some(NameVal::from_meta(meta).ok()?.value),
        }
    }
}

pub struct RemoveHelpers;

impl VisitMut for RemoveHelpers {
    fn visit_fn_arg_mut(&mut self, i: &mut syn::FnArg) {
        match i {
            syn::FnArg::Receiver(recv) => recv
                .attrs
                .retain(|it| !is_comptime_attr(it) && !is_define_attribute(it)),
            syn::FnArg::Typed(typed) => typed
                .attrs
                .retain(|it| !is_comptime_attr(it) && !is_define_attribute(it)),
        }
        visit_mut::visit_fn_arg_mut(self, i);
    }

    fn visit_expr_for_loop_mut(&mut self, i: &mut syn::ExprForLoop) {
        let unroll = Unroll::unroll_expr(&i.attrs);
        i.attrs.retain(|attr| !is_unroll_attr(attr));
        if let Some(unroll) = unroll {
            i.body
                .stmts
                .insert(0, parse_quote![let __unroll = #unroll;])
        }
        visit_mut::visit_expr_for_loop_mut(self, i);
    }

    fn visit_local_mut(&mut self, i: &mut syn::Local) {
        i.attrs.retain(|attr| !is_comptime_attr(attr));
        visit_mut::visit_local_mut(self, i);
    }

    fn visit_expr_match_mut(&mut self, i: &mut syn::ExprMatch) {
        i.attrs.retain(|attr| !is_comptime_attr(attr));
        visit_mut::visit_expr_match_mut(self, i);
    }

    fn visit_expr_if_mut(&mut self, i: &mut syn::ExprIf) {
        i.attrs.retain(|attr| !is_comptime_attr(attr));
        visit_mut::visit_expr_if_mut(self, i);
    }

    fn visit_type_param_mut(&mut self, i: &mut syn::TypeParam) {
        i.attrs.retain(|attr| !is_helper(attr));
        visit_mut::visit_type_param_mut(self, i);
    }
}

pub struct ReplaceDefines;

impl VisitMut for ReplaceDefines {
    fn visit_block_mut(&mut self, i: &mut syn::Block) {
        let stmts = core::mem::take(&mut i.stmts);
        i.stmts = stmts
            .into_iter()
            .flat_map(|stmt| match stmt {
                Stmt::Local(local) => {
                    if let Some((name, kind, init)) = parse_define_macro(&local) {
                        let define: Stmt = match kind {
                            DefineKind::Type => {
                                let define_size = prelude_type("define_scalar");
                                parse_quote![#define_size!(#name);]
                            }
                            DefineKind::Size => {
                                let define_size = prelude_type("define_size");
                                parse_quote![#define_size!(#name);]
                            }
                        };
                        let init: Stmt = parse_quote!(let _ = #init;);
                        vec![define, init]
                    } else {
                        vec![Stmt::Local(local)]
                    }
                }
                other => vec![other],
            })
            .collect();
        visit_mut::visit_block_mut(self, i);
    }
}

pub fn is_comptime_attr(attr: &Attribute) -> bool {
    attr.path().is_ident("comptime")
}

pub fn is_unroll_attr(attr: &Attribute) -> bool {
    attr.path().is_ident("unroll")
}

pub fn is_expr_attribute(attr: &Attribute) -> bool {
    attr.path().is_ident("expr")
}

pub fn is_define_attribute(attr: &Attribute) -> bool {
    attr.path().is_ident("define")
}

pub fn is_helper(attr: &Attribute) -> bool {
    is_comptime_attr(attr)
        || is_unroll_attr(attr)
        || is_expr_attribute(attr)
        || is_define_attribute(attr)
}
