use quote::{ToTokens, format_ident, quote};
use syn::{
    FnArg, GenericArgument, Generics, Ident, ImplItem, ItemImpl, PathArguments, Token, Type,
    TypePath, spanned::Spanned, visit_mut::VisitMut,
};

use crate::{
    parse::kernel::{KernelArgs, KernelBody},
    scope::Context,
};

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
    MethodExpand(KernelFn),
    FnExpand(KernelFn),
    Other,
}

impl CubeImplItem {
    pub fn from_impl_item(
        struct_ty_name: &Type,
        item: ImplItem,
        args: &KernelArgs,
    ) -> syn::Result<Vec<Self>> {
        let res = match item {
            ImplItem::Fn(func) => {
                let name = func.sig.ident.clone();
                let full_name = quote!(#struct_ty_name::#name).to_string();

                let is_method = func
                    .sig
                    .inputs
                    .iter()
                    .any(|param| matches!(param, FnArg::Receiver(_)));
                let func_name_expand = format_ident!("__expand_{}", func.sig.ident);
                let mut func =
                    KernelFn::from_sig_and_block(func.vis, func.sig, func.block, full_name, args)?;

                if is_method {
                    let method = Self::handle_method_expand(func_name_expand, &mut func);
                    let func_expand = Self::create_func_expand(struct_ty_name, &func, true);

                    vec![
                        CubeImplItem::Fn(func),
                        CubeImplItem::MethodExpand(method),
                        CubeImplItem::FnExpand(func_expand),
                    ]
                } else {
                    func.sig.name = func_name_expand;

                    let func_expand = Self::create_func_expand(struct_ty_name, &func, false);
                    vec![CubeImplItem::Fn(func), CubeImplItem::FnExpand(func_expand)]
                }
            }
            _ => vec![CubeImplItem::Other],
        };

        Ok(res)
    }

    pub fn as_func(&mut self) -> Option<&mut KernelFn> {
        match self {
            CubeImplItem::Fn(func) => Some(func),
            _ => None,
        }
    }

    pub fn as_func_expand(&mut self) -> Option<&mut KernelFn> {
        match self {
            CubeImplItem::FnExpand(func) => Some(func),
            _ => None,
        }
    }

    pub fn as_method_expand(&mut self) -> Option<&mut KernelFn> {
        match self {
            CubeImplItem::MethodExpand(func) => Some(func),
            _ => None,
        }
    }

    /// Create the method from the function and update the function body to
    /// point to the method's implementation.
    fn handle_method_expand(func_name_expand: Ident, func: &mut KernelFn) -> KernelFn {
        let mut method_sig = func.sig.clone();

        method_sig.name = format_ident!("__expand_{}_method", func.sig.name);
        method_sig.plain_returns_self();

        // Since the function is associated to the expand type, we have to update the
        // normalized types for the arguments.
        for param in method_sig
            .parameters
            .iter_mut()
            // We skip the self param.
            .skip(1)
        {
            param.plain_normalized_self();
        }

        func.sig.name = func_name_expand;
        func.sig.receiver_arg = None;
        let param = func.sig.parameters.first_mut().expect("Should be a method");
        param.name = Ident::new("this", param.span());

        let args = func.sig.parameters.iter().skip(1).map(|param| &param.name);
        let method_name = &method_sig.name;
        let (_, generics, _) = &method_sig.generics.split_for_impl();
        let generics = generics.as_turbofish();

        let mut body = KernelBody::Verbatim(quote! {
            this.#method_name #generics(
                scope,
                #(#args),*
            )
        });

        // The function points to the method's body.
        core::mem::swap(&mut func.body, &mut body);

        KernelFn {
            vis: func.vis.clone(),
            sig: method_sig,
            body,
            full_name: func.full_name.clone(),
            span: func.span,
            context: Context::new(func.context.return_type.clone(), func.debug_symbols),
            src_file: func.src_file.clone(),
            debug_symbols: func.debug_symbols,
        }
    }

    /// Create the same function but that should be generated for the expand
    /// type.
    ///
    /// This is important since it allows to use the Self keyword inside
    /// methods.
    fn create_func_expand(struct_ty_name: &Type, func: &KernelFn, is_method: bool) -> KernelFn {
        let mut func_sig = func.sig.clone();

        // Since the function is associated to the expand type, we have to update the
        // normalized types for the arguments.
        for param in func_sig
            .parameters
            .iter_mut()
            // We skip the self param.
            .skip(if is_method { 1 } else { 0 })
        {
            param.plain_normalized_self();
        }

        if let Some(param) = func_sig.parameters.first_mut()
            && is_method
        {
            let ty = match &param.ty {
                Type::Reference(reference) => reference.elem.as_ref().clone(),
                ty => ty.clone(),
            };
            param.name = Ident::new("this", param.span());
            param.normalized_ty = ty;
            func_sig.receiver_arg = None;
        }
        func_sig.plain_returns_self();

        let args = func_sig.parameters.iter().map(|param| &param.name);
        let struct_name = format_type_with_turbofish(struct_ty_name);
        let fn_name = &func_sig.name;
        let (_, generics, _) = &func_sig.generics.split_for_impl();
        let generics = generics.as_turbofish();

        let body = quote! {
            #struct_name::#fn_name #generics(
                scope,
                #(#args),*
            )
        };

        KernelFn {
            vis: func.vis.clone(),
            sig: func_sig,
            body: KernelBody::Verbatim(body),
            full_name: func.full_name.clone(),
            span: func.span,
            context: Context::new(func.context.return_type.clone(), func.debug_symbols),
            src_file: func.src_file.clone(),
            debug_symbols: func.debug_symbols,
        }
    }
}

impl CubeImpl {
    pub fn from_item_impl(mut item_impl: ItemImpl, args: &KernelArgs) -> syn::Result<Self> {
        let struct_name = *item_impl.self_ty.clone();

        let items = item_impl
            .items
            .iter()
            .cloned()
            .map(|item| CubeImplItem::from_impl_item(&struct_name, item, args))
            .flat_map(|items| {
                let result: Vec<syn::Result<CubeImplItem>> = match items {
                    Ok(items) => items.into_iter().map(Ok).collect(),
                    Err(err) => vec![Err(err)],
                };
                result
            })
            .collect::<Result<_, _>>()?;

        RemoveHelpers.visit_item_impl_mut(&mut item_impl);
        ReplaceIndices.visit_item_impl_mut(&mut item_impl);

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

/// When we use a type with generics for calling a function, we have to add more
/// `::` between the type ident and the generic arguments.
fn format_type_with_turbofish(ty: &Type) -> proc_macro2::TokenStream {
    match ty {
        Type::Path(TypePath { path, .. }) => {
            let segments = &path.segments;
            let last_segment = segments.last().unwrap();
            let ident = &last_segment.ident;

            match &last_segment.arguments {
                PathArguments::AngleBracketed(args) => {
                    let generic_args = args.args.iter().map(|arg| match arg {
                        GenericArgument::Type(t) => t.to_token_stream(),
                        _ => quote! { #arg },
                    });

                    quote! { #ident::<#(#generic_args),*> }
                }
                _ => quote! { #ident },
            }
        }
        _ => ty.to_token_stream(),
    }
}
