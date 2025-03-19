use quote::{format_ident, quote};
use syn::{
    Attribute, Generics, Ident, ImplItem, ItemImpl, ItemTrait, LitStr, Path, Token, TraitItem,
    Type, Visibility, visit_mut::VisitMut,
};

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
    Other,
}

pub enum CubeTraitImplItem {
    Fn(KernelFn),
    Other,
}

impl CubeTraitItem {
    pub fn from_trait_item(item: TraitItem) -> syn::Result<Self> {
        let res = match item {
            TraitItem::Fn(func) => {
                let mut func = KernelSignature::from_trait_fn(func)?;
                func.name = format_ident!("__expand_{}", func.name);
                CubeTraitItem::Fn(func)
            }
            _ => CubeTraitItem::Other,
        };
        Ok(res)
    }

    pub fn func(&self) -> Option<&KernelSignature> {
        match self {
            CubeTraitItem::Fn(func) => Some(func),
            CubeTraitItem::Other => None,
        }
    }
}

impl CubeTraitImplItem {
    pub fn from_impl_item(
        struct_ty: &Type,
        item: ImplItem,
        src_file: Option<LitStr>,
    ) -> syn::Result<Self> {
        let res = match item {
            ImplItem::Fn(func) => {
                let name = func.sig.ident.clone();
                let full_name = quote!(#struct_ty::#name).to_string();
                let mut func = KernelFn::from_sig_and_block(
                    func.vis, func.sig, func.block, full_name, src_file,
                )?;
                func.sig.name = format_ident!("__expand_{}", func.sig.name);
                CubeTraitImplItem::Fn(func)
            }
            _ => CubeTraitImplItem::Other,
        };
        Ok(res)
    }

    pub fn func(&mut self) -> Option<&mut KernelFn> {
        match self {
            CubeTraitImplItem::Fn(func) => Some(func),
            CubeTraitImplItem::Other => None,
        }
    }
}

impl CubeTrait {
    pub fn from_item_trait(item: ItemTrait) -> syn::Result<Self> {
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
            .into_iter()
            .map(CubeTraitItem::from_trait_item)
            .collect::<Result<_, _>>()?;

        Ok(Self {
            attrs,
            vis,
            unsafety,
            name,
            generics,
            items,
            original_trait,
        })
    }
}

impl CubeTraitImpl {
    pub fn from_item_impl(mut item_impl: ItemImpl, src_file: Option<LitStr>) -> syn::Result<Self> {
        let items = item_impl
            .items
            .iter()
            .cloned()
            .map(|item| {
                CubeTraitImplItem::from_impl_item(&item_impl.self_ty, item, src_file.clone())
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
