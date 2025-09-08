use proc_macro2::TokenStream;
use quote::{ToTokens, format_ident, quote};
use syn::{
    Attribute, FnArg, Generics, Ident, ImplItem, ItemImpl, ItemTrait, LitStr, Path, Signature,
    Token, TraitItem, Type, TypeParamBound, Visibility, punctuated::Punctuated,
    visit_mut::VisitMut,
};

use crate::parse::kernel::KernelArgs;

use super::{
    StripBounds, StripDefault,
    helpers::{RemoveHelpers, ReplaceIndices},
    kernel::{KernelFn, KernelSignature},
};

pub struct CubeTrait {
    pub attrs: Vec<Attribute>,
    pub vis: Visibility,
    pub unsafety: Option<Token![unsafe]>,
    pub name: Ident,
    pub generics: Generics,
    pub items: Vec<CubeTraitItem>,
    pub original_trait: ItemTrait,
    pub expand_supertraits: Punctuated<TypeParamBound, Token![+]>,
}

pub struct CubeTraitImpl {
    pub unsafety: Option<Token![unsafe]>,
    pub struct_name: Type,
    pub trait_name: Path,
    pub generics: Generics,
    pub items: Vec<CubeTraitImplItem>,
    pub original_items: Vec<ImplItem>,
}

pub enum CubeTraitItem {
    Fn(KernelSignature),
    Method(KernelSignature),
    Other(TokenStream),
}

pub enum CubeTraitImplItem {
    Fn(KernelFn),
    Method(KernelFn),
    Other(TokenStream),
}

impl CubeTraitItem {
    pub fn from_trait_item(item: TraitItem) -> syn::Result<Self> {
        let res = match item {
            TraitItem::Fn(func) if has_receiver(&func.sig) => {
                let mut func = KernelSignature::from_trait_fn(func)?;
                func.name = format_ident!("__expand_{}_method", func.name);
                CubeTraitItem::Method(func)
            }
            TraitItem::Fn(func) => {
                let mut func = KernelSignature::from_trait_fn(func)?;
                func.name = format_ident!("__expand_{}", func.name);
                CubeTraitItem::Fn(func)
            }
            other => CubeTraitItem::Other(other.to_token_stream()),
        };
        Ok(res)
    }

    pub fn func(&self) -> Option<&KernelSignature> {
        match self {
            CubeTraitItem::Fn(func) => Some(func),
            CubeTraitItem::Method(_) | CubeTraitItem::Other(_) => None,
        }
    }

    pub fn method(&self) -> Option<&KernelSignature> {
        match self {
            CubeTraitItem::Method(method) => Some(method),
            CubeTraitItem::Fn(_) | CubeTraitItem::Other(_) => None,
        }
    }

    pub fn other(&self) -> Option<&TokenStream> {
        match self {
            CubeTraitItem::Fn(_) | CubeTraitItem::Method(_) => None,
            CubeTraitItem::Other(tokens) => Some(tokens),
        }
    }
}

impl CubeTraitImplItem {
    pub fn from_impl_item(
        struct_ty: &Type,
        item: ImplItem,
        src_file: Option<LitStr>,
        debug_symbols: bool,
    ) -> syn::Result<Self> {
        let res = match item {
            ImplItem::Fn(func) => {
                let is_method = has_receiver(&func.sig);
                let name = func.sig.ident.clone();
                let full_name = quote!(#struct_ty::#name).to_string();

                let mut func = KernelFn::from_sig_and_block(
                    func.vis,
                    func.sig,
                    func.block,
                    full_name,
                    src_file,
                    debug_symbols,
                )?;
                if is_method {
                    func.sig.name = format_ident!("__expand_{}_method", func.sig.name);
                    CubeTraitImplItem::Method(func)
                } else {
                    func.sig.name = format_ident!("__expand_{}", func.sig.name);
                    CubeTraitImplItem::Fn(func)
                }
            }
            other => CubeTraitImplItem::Other(other.to_token_stream()),
        };
        Ok(res)
    }

    pub fn func(&mut self) -> Option<&mut KernelFn> {
        match self {
            CubeTraitImplItem::Fn(func) => Some(func),
            CubeTraitImplItem::Method(_) | CubeTraitImplItem::Other(_) => None,
        }
    }

    pub fn method(&mut self) -> Option<&mut KernelFn> {
        match self {
            CubeTraitImplItem::Method(method) => Some(method),
            CubeTraitImplItem::Fn(_) | CubeTraitImplItem::Other(_) => None,
        }
    }

    pub fn other(&self) -> Option<&TokenStream> {
        match self {
            CubeTraitImplItem::Other(tokens) => Some(tokens),
            CubeTraitImplItem::Fn(_) | CubeTraitImplItem::Method(_) => None,
        }
    }
}

impl CubeTrait {
    pub fn from_item_trait(item: ItemTrait, args: KernelArgs) -> syn::Result<Self> {
        let mut original_trait = item.clone();
        RemoveHelpers.visit_item_trait_mut(&mut original_trait);

        let mut attrs = item.attrs;
        attrs.retain(|attr| !attr.path().is_ident("cube"));
        attrs.retain(|attr| !attr.path().is_ident("cube"));
        let vis = item.vis;
        let unsafety = item.unsafety;
        let name = item.ident;

        let mut original_generic_names = item.generics.clone();
        StripBounds.visit_generics_mut(&mut original_generic_names);

        let mut generics = item.generics;
        StripDefault.visit_generics_mut(&mut generics);

        let items = item
            .items
            .clone()
            .into_iter()
            .map(CubeTraitItem::from_trait_item)
            .collect::<Result<_, _>>()?;
        let mut expand_supertraits = Punctuated::new();
        if let Some(base_traits) = args.expand_base_traits {
            for base_trait in base_traits.split(",") {
                let bound: TypeParamBound = syn::parse_str(base_trait.trim())?;
                expand_supertraits.push(bound);
            }
        }

        Ok(Self {
            attrs,
            vis,
            unsafety,
            name,
            generics,
            items,
            original_trait,
            expand_supertraits,
        })
    }
}

fn has_receiver(sig: &Signature) -> bool {
    sig.inputs.iter().any(|it| matches!(it, FnArg::Receiver(_)))
}

impl CubeTraitImpl {
    pub fn from_item_impl(
        mut item_impl: ItemImpl,
        src_file: Option<LitStr>,
        debug_symbols: bool,
    ) -> syn::Result<Self> {
        let items = item_impl
            .items
            .iter()
            .cloned()
            .map(|item| {
                CubeTraitImplItem::from_impl_item(
                    &item_impl.self_ty,
                    item,
                    src_file.clone(),
                    debug_symbols,
                )
            })
            .collect::<Result<_, _>>()?;

        RemoveHelpers.visit_item_impl_mut(&mut item_impl);
        ReplaceIndices.visit_item_impl_mut(&mut item_impl);

        let struct_name = *item_impl.self_ty;
        let trait_name = item_impl.trait_.unwrap().1;

        let mut attrs = item_impl.attrs;
        attrs.retain(|attr| !attr.path().is_ident("cube"));
        let unsafety = item_impl.unsafety;

        let generics = item_impl.generics;

        Ok(Self {
            unsafety,
            struct_name,
            trait_name,
            generics,
            items,
            original_items: item_impl.items,
        })
    }
}
