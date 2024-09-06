use darling::usage::{GenericsExt, Purpose, UsesLifetimes, UsesTypeParams};
use proc_macro2::TokenStream;
use quote::{format_ident, ToTokens};
use syn::{
    parse_quote, visit_mut::VisitMut, Attribute, Generics, Ident, ImplItem, ItemImpl, ItemTrait,
    Path, Token, TraitItem, Visibility,
};

use crate::paths::frontend_type;

use super::{
    helpers::RemoveHelpers,
    kernel::{CubeTraitArgs, CubeTraitImplArgs, KernelFn, KernelSignature},
    StripBounds, StripDefault,
};

pub struct CubeTrait {
    pub attrs: Vec<Attribute>,
    pub vis: Visibility,
    pub unsafety: Option<Token![unsafe]>,
    pub expand_name: Ident,
    pub generics: Generics,
    pub items: Vec<CubeTraitItem>,
    pub original_trait: ItemTrait,
}

pub struct CubeTraitImpl {
    pub unsafety: Option<Token![unsafe]>,
    pub struct_expand_name: Ident,
    pub struct_generics: Generics,
    pub trait_expand_name: Path,
    pub generics: Generics,
    pub items: Vec<CubeTraitImplItem>,
}

pub enum CubeTraitItem {
    Fn(KernelSignature),
    Other(TokenStream),
}

pub enum CubeTraitImplItem {
    Fn(KernelFn),
    Other(TokenStream),
}

impl CubeTraitItem {
    pub fn from_trait_item(item: TraitItem) -> syn::Result<Self> {
        let res = match item {
            TraitItem::Fn(func) => CubeTraitItem::Fn(KernelSignature::from_trait_fn(func)?),
            other => CubeTraitItem::Other(other.to_token_stream()),
        };
        Ok(res)
    }
}

impl CubeTraitImplItem {
    pub fn from_impl_item(item: ImplItem) -> syn::Result<Self> {
        let res = match item {
            ImplItem::Fn(func) => {
                CubeTraitImplItem::Fn(KernelFn::from_sig_and_block(func.sig, func.block, false)?)
            }
            other => CubeTraitImplItem::Other(other.to_token_stream()),
        };
        Ok(res)
    }
}

impl CubeTrait {
    pub fn from_item_trait(item: ItemTrait, args: CubeTraitArgs) -> syn::Result<Self> {
        let static_expand = frontend_type("StaticExpand");
        let mut original_trait = item.clone();
        RemoveHelpers.visit_item_trait_mut(&mut original_trait);

        let mut attrs = item.attrs;
        attrs.retain(|attr| !attr.path().is_ident("cube"));
        attrs.retain(|attr| !attr.path().is_ident("cube"));
        let vis = item.vis;
        let unsafety = item.unsafety;
        let name = item.ident;
        let expand_name = args
            .expand_name
            .unwrap_or_else(|| format_ident!("{name}Expand"));

        let mut original_generic_names = item.generics.clone();
        StripBounds.visit_generics_mut(&mut original_generic_names);

        let mut generics = item.generics;
        StripDefault.visit_generics_mut(&mut generics);

        let items = item
            .items
            .into_iter()
            .map(CubeTraitItem::from_trait_item)
            .collect::<Result<_, _>>()?;

        original_trait
            .supertraits
            .push(parse_quote![#static_expand<Expanded: #expand_name #original_generic_names>]);

        Ok(Self {
            attrs,
            vis,
            unsafety,
            expand_name,
            generics,
            items,
            original_trait,
        })
    }
}

impl CubeTraitImpl {
    pub fn from_item_impl(item_impl: ItemImpl, args: CubeTraitImplArgs) -> syn::Result<Self> {
        let struct_name = *item_impl.self_ty;
        let struct_name: Path = parse_quote![#struct_name];
        let struct_expand_name = args.expand_name.unwrap_or_else(|| {
            format_ident!(
                "{}Expand",
                struct_name.segments.last().cloned().unwrap().ident
            )
        });
        let trait_name = item_impl.trait_.unwrap().1;
        let trait_expand_name = args.trait_expand_name.unwrap_or_else(|| {
            let mut path = trait_name.clone();
            let last = path.segments.last_mut().unwrap();
            last.ident = format_ident!("{}Expand", last.ident);
            path
        });

        let mut attrs = item_impl.attrs;
        attrs.retain(|attr| !attr.path().is_ident("cube"));
        attrs.retain(|attr| !attr.path().is_ident("cube"));
        let unsafety = item_impl.unsafety;

        let generics = item_impl.generics;
        let mut generic_names = generics.clone();
        StripBounds.visit_generics_mut(&mut generic_names);

        let struct_generic_names = struct_name.segments.last().unwrap().arguments.clone();
        let lifetimes = generics.declared_lifetimes();
        let type_params = generics.declared_type_params();

        let struct_generic_opts = Purpose::Declare.into();
        let struct_lifetimes =
            struct_generic_names.uses_lifetimes_cloned(&struct_generic_opts, &lifetimes);
        let struct_type_params =
            struct_generic_names.uses_type_params_cloned(&struct_generic_opts, &type_params);
        let struct_generics = if struct_lifetimes.is_empty() && struct_type_params.is_empty() {
            Generics::default()
        } else {
            let lifetimes = struct_lifetimes.into_iter();
            let types = struct_type_params.into_iter();
            parse_quote![<#(#lifetimes,)* #(#types),*>]
        };

        let items = item_impl
            .items
            .into_iter()
            .map(CubeTraitImplItem::from_impl_item)
            .collect::<Result<_, _>>()?;

        Ok(Self {
            unsafety,
            struct_expand_name,
            struct_generics,
            trait_expand_name,
            generics,
            items,
        })
    }
}
