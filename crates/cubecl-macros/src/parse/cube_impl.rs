use quote::format_ident;
use syn::{visit_mut::VisitMut, Generics, ImplItem, ItemImpl, Token, Type};

use super::{
    helpers::{RemoveHelpers, ReplaceIndices},
    kernel::KernelFn,
};

pub struct CubeImpl {
    pub unsafety: Option<Token![unsafe]>,
    pub struct_name: Type,
    pub generics: Generics,
    pub items: Vec<CubeImplItem>,
    pub original_items: Vec<ImplItem>,
}

pub enum CubeImplItem {
    Fn(KernelFn),
    Method(KernelFn),
    Other,
}

impl CubeImplItem {
    pub fn from_impl_item(item: ImplItem) -> syn::Result<Vec<Self>> {
        let res = match item {
            ImplItem::Fn(func) => {
                let mut func = KernelFn::from_sig_and_block(func.sig, func.block)?;
                if func.sig.parameters.first().map(|param| param.name == "self").unwrap_or(false) {
                    let sig = func.sig.clone();

                }
                func.sig.name = format_ident!("__expand_{}", func.sig.name);
                CubeImplItem::Fn(func)
            }
            _ => CubeImplItem::Other,
        };
        Ok(res)
    }

    pub fn func(&mut self) -> Option<&mut KernelFn> {
        match self {
            CubeImplItem::Fn(func) => Some(func),
            CubeImplItem::Method(func) => None,
            CubeImplItem::Other => None,
        }
    }
}

impl CubeImpl {
    pub fn from_item_impl(mut item_impl: ItemImpl) -> syn::Result<Self> {
        let items = item_impl
            .items
            .iter()
            .cloned()
            .map(CubeImplItem::from_impl_item)
            .collect::<Result<_, _>>()?;

        RemoveHelpers.visit_item_impl_mut(&mut item_impl);
        ReplaceIndices.visit_item_impl_mut(&mut item_impl);

        let struct_name = *item_impl.self_ty;

        let mut attrs = item_impl.attrs;
        attrs.retain(|attr| !attr.path().is_ident("cube"));

        let unsafety = item_impl.unsafety;
        let generics = item_impl.generics;

        Ok(Self {
            unsafety,
            struct_name,
            generics,
            items,
            original_items: item_impl.items,
        })
    }
}
