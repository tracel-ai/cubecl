use darling::FromMeta;
use syn::{
    Attribute, Expr, ExprReference, parse_quote,
    visit_mut::{self, VisitMut},
};

use crate::{expression::Expression, paths::prelude_path, scope::Context};

pub struct Unroll {
    pub value: Expression,
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
            },
            syn::Meta::List(list) => {
                let expr = syn::parse2(list.tokens.clone())?;
                let expr = Expression::from_expr(expr, context)?;
                Self { value: expr }
            }
            meta => {
                let expr = NameVal::from_meta(meta)?;
                let expr = Expression::from_expr(expr.value, context)?;
                Self { value: expr }
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
            syn::FnArg::Receiver(recv) => recv.attrs.retain(|it| !is_comptime_attr(it)),
            syn::FnArg::Typed(typed) => typed.attrs.retain(|it| !is_comptime_attr(it)),
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
}

pub struct ReplaceIndices;
pub struct ReplaceIndex;
pub struct ReplaceIndexMut;

impl VisitMut for ReplaceIndices {
    fn visit_expr_assign_mut(&mut self, i: &mut syn::ExprAssign) {
        ReplaceIndexMut.visit_expr_mut(&mut i.left);
        ReplaceIndex.visit_expr_mut(&mut i.right);
        visit_mut::visit_expr_assign_mut(self, i);
    }

    fn visit_expr_binary_mut(&mut self, i: &mut syn::ExprBinary) {
        match i.op {
            syn::BinOp::AddAssign(_)
            | syn::BinOp::SubAssign(_)
            | syn::BinOp::MulAssign(_)
            | syn::BinOp::DivAssign(_)
            | syn::BinOp::RemAssign(_)
            | syn::BinOp::BitXorAssign(_)
            | syn::BinOp::BitAndAssign(_)
            | syn::BinOp::BitOrAssign(_)
            | syn::BinOp::ShlAssign(_)
            | syn::BinOp::ShrAssign(_) => {
                ReplaceIndexMut.visit_expr_mut(&mut i.left);
                ReplaceIndex.visit_expr_mut(&mut i.right);
            }
            _ => {}
        }
        visit_mut::visit_expr_binary_mut(self, i);
    }

    fn visit_expr_mut(&mut self, i: &mut syn::Expr) {
        match i {
            Expr::Reference(ExprReference {
                mutability: Some(_),
                expr,
                ..
            }) => {
                ReplaceIndexMut.visit_expr_mut(expr);
            }
            Expr::Index(_) => ReplaceIndex.visit_expr_mut(i),
            _ => {}
        }
        visit_mut::visit_expr_mut(self, i);
    }

    fn visit_item_fn_mut(&mut self, i: &mut syn::ItemFn) {
        let prelude_path = prelude_path();
        let import = parse_quote![use #prelude_path::{CubeIndex as _, CubeIndexMut as _};];
        i.block.stmts.insert(0, import);
        visit_mut::visit_item_fn_mut(self, i);
    }

    fn visit_impl_item_fn_mut(&mut self, i: &mut syn::ImplItemFn) {
        let prelude_path = prelude_path();
        let import = parse_quote![use #prelude_path::{CubeIndex as _, CubeIndexMut as _};];
        i.block.stmts.insert(0, import);
        visit_mut::visit_impl_item_fn_mut(self, i);
    }

    fn visit_trait_item_fn_mut(&mut self, i: &mut syn::TraitItemFn) {
        if let Some(block) = &mut i.default {
            let prelude_path = prelude_path();
            let import = parse_quote![use #prelude_path::{CubeIndex as _, CubeIndexMut as _};];
            block.stmts.insert(0, import);
        }
        visit_mut::visit_trait_item_fn_mut(self, i);
    }
}

impl VisitMut for ReplaceIndex {
    fn visit_expr_mut(&mut self, i: &mut Expr) {
        match i {
            Expr::Reference(ExprReference {
                mutability: Some(_),
                expr,
                ..
            }) => {
                ReplaceIndexMut.visit_expr_mut(expr);
            }
            Expr::Index(index) => {
                let inner = &index.expr;
                let index = &index.index;
                *i = parse_quote![*#inner.cube_idx(#index)]
            }
            _ => {}
        }
        visit_mut::visit_expr_mut(self, i);
    }
}

impl VisitMut for ReplaceIndexMut {
    fn visit_expr_mut(&mut self, i: &mut syn::Expr) {
        if let Expr::Index(index) = i {
            let inner = &index.expr;
            let index = &index.index;
            *i = parse_quote![*#inner.cube_idx_mut(#index)]
        }
        visit_mut::visit_expr_mut(self, i);
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

pub fn is_helper(attr: &Attribute) -> bool {
    is_comptime_attr(attr) || is_unroll_attr(attr) || is_expr_attribute(attr)
}
