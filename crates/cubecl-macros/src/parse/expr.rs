use darling::{
    usage::{CollectLifetimes, CollectTypeParams, GenericsExt, Purpose},
    util::Flag,
    FromAttributes, FromMeta,
};
use ident_case::RenameRule;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse_quote, spanned::Spanned, visit_mut::VisitMut as _, Expr, FnArg, Generics, Ident, ItemFn,
    Pat, PatType, Type, Visibility,
};

use super::helpers::RemoveHelpers;

#[derive(FromMeta)]
pub struct ExpressionArgs {
    pub name: Option<Ident>,
    pub vectorization: Option<Expr>,
    pub output: Expr,
}

#[derive(FromAttributes)]
#[darling(attributes(expr))]
pub struct ExprAttribute {
    pub comptime: Flag,
    pub inner: Flag,
}

pub struct Expression {
    pub vis: Visibility,
    pub generics: Generics,
    pub name: Ident,
    pub args: Vec<ExpressionArg>,
    pub phantom_generics: Option<TokenStream>,
    pub output: Expr,
    pub item: ItemFn,
    pub vectorization: Expr,
}

pub struct ExpressionArg {
    pub name: Pat,
    pub ty: Type,
    pub _comptime: bool,
    pub inner: bool,
}

impl Expression {
    pub fn from_item_fn(mut item: ItemFn, params: ExpressionArgs) -> syn::Result<Self> {
        let struct_name = params.name.unwrap_or_else(|| {
            let casing = RenameRule::PascalCase.apply_to_field(item.sig.ident.to_string());
            format_ident!("{casing}")
        });

        let lifetimes = item.sig.generics.declared_lifetimes();
        let type_params = item.sig.generics.declared_type_params();

        let types = item
            .sig
            .inputs
            .iter()
            .map(unwrap_fn_arg)
            .map(|arg| *arg.ty.clone())
            .collect::<Vec<_>>();
        let used_lifetimes = types
            .iter()
            .take(types.len() - 1)
            .collect_lifetimes_cloned(&Purpose::Declare.into(), &lifetimes);
        let used_type_params = types
            .iter()
            .take(types.len() - 1)
            .collect_type_params_cloned(&Purpose::Declare.into(), &type_params);

        let unused_lifetimes: Vec<_> = lifetimes.difference(&used_lifetimes).collect();
        let unused_type_params: Vec<_> = type_params.difference(&used_type_params).collect();
        let has_unused = !unused_lifetimes.is_empty() || !unused_type_params.is_empty();
        let phantom_generics =
            has_unused.then(|| quote![::core::marker::PhantomData<(#(#unused_lifetimes,)* #(#unused_type_params),*)>]);

        let mut args = item
            .sig
            .inputs
            .iter()
            .map(unwrap_fn_arg)
            .map(ExpressionArg::from_pat_ty)
            .collect::<Vec<_>>();
        args.pop();
        if args.iter().filter(|it| it.inner).count() > 1 {
            Err(syn::Error::new(
                item.span(),
                "Can't have more than one forwarded parameter",
            ))?;
        }

        RemoveHelpers.visit_item_fn_mut(&mut item);
        let inner_fn = item.clone();
        let vis = item.vis;
        let generics = item.sig.generics;
        let vectorization = params
            .vectorization
            .or_else(|| {
                let inner = &args.iter().find(|it| it.inner)?.name;
                Some(parse_quote![self.#inner.vectorization()])
            })
            .unwrap_or_else(|| parse_quote![None]);

        Ok(Self {
            vis,
            generics,
            name: struct_name,
            phantom_generics,
            args,
            output: params.output,
            item: inner_fn,
            vectorization,
        })
    }
}

impl ExpressionArg {
    pub fn from_pat_ty(pat_ty: &PatType) -> Self {
        let attr = ExprAttribute::from_attributes(&pat_ty.attrs).ok();
        let name = &pat_ty.pat;
        let ty = match &*pat_ty.ty {
            Type::Reference(reference) => &*reference.elem,
            ty => ty,
        };
        let comptime = attr
            .as_ref()
            .map(|it| it.comptime.is_present())
            .unwrap_or(false);
        let inner = attr
            .as_ref()
            .map(|it| it.inner.is_present())
            .unwrap_or(false);

        Self {
            name: *name.clone(),
            ty: ty.clone(),
            _comptime: comptime,
            inner,
        }
    }
}

fn unwrap_fn_arg(arg: &FnArg) -> &PatType {
    match arg {
        FnArg::Receiver(_) => panic!("Receiver not supported"),
        FnArg::Typed(typed) => typed,
    }
}
