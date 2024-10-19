use quote::{format_ident, quote};
use syn::{
    spanned::Spanned, visit_mut::VisitMut, Generics, Ident, ImplItem, ItemImpl, Token, Type,
};

use crate::parse::kernel::KernelBody;

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
                if func
                    .sig
                    .parameters
                    .first()
                    .map(|param| param.name == "self")
                    .unwrap_or(false)
                {
                    let mut method = func.clone();

                    method.sig.name = format_ident!("__expand_{}_method", func.sig.name);
                    func.sig.name = format_ident!("__expand_{}", func.sig.name);

                    if let Some(param) = func.sig.parameters.first_mut() {
                        param.name = Ident::new("this".into(), param.span());

                        let args = func.sig.parameters.iter().skip(1).map(|param| &param.name);
                        let fna = &method.sig.name;

                        let tokens = quote! {
                            this.#fna(
                                context,
                                #(#args),*
                            )
                        };
                        func.body = KernelBody::Verbatim(tokens);
                    }

                    return Ok(vec![CubeImplItem::Fn(func), CubeImplItem::Method(method)]);
                } else {
                    func.sig.name = format_ident!("__expand_{}", func.sig.name);
                }
                CubeImplItem::Fn(func)
            }
            _ => CubeImplItem::Other,
        };
        Ok(vec![res])
    }

    pub fn func(&mut self) -> Option<&mut KernelFn> {
        match self {
            CubeImplItem::Fn(func) => Some(func),
            _ => None,
        }
    }

    pub fn method(&mut self) -> Option<&mut KernelFn> {
        match self {
            CubeImplItem::Method(func) => Some(func),
            _ => None,
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
            .map(|items| {
                let result: Vec<syn::Result<CubeImplItem>> = match items {
                    Ok(items) => items.into_iter().map(|item| Ok(item)).collect(),
                    Err(err) => vec![Err(err)],
                };
                result
            })
            .flatten()
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
