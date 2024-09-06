use std::iter;

use darling::usage::{CollectLifetimes as _, CollectTypeParams as _, GenericsExt as _, Purpose};
use proc_macro2::TokenStream;
use quote::{format_ident, quote, quote_spanned, ToTokens};

use crate::{
    parse::kernel::{KernelFn, KernelParam, KernelSignature, Launch},
    paths::{core_type, prelude_type},
};

impl ToTokens for KernelFn {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let sig = &self.sig;
        let block = &self.block;

        let out = quote! {
            #sig {
                #block
            }
        };
        tokens.extend(out);
    }
}

impl ToTokens for KernelSignature {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let cube_context = prelude_type("CubeContext");
        let cube_type = prelude_type("CubeType");

        let name = &self.name;
        let generics = &self.generics;
        let return_type = &self.returns;
        let args = &self.parameters;

        let out = quote! {
            fn #name #generics(
                context: &mut #cube_context,
                #(#args),*
            ) -> <#return_type as #cube_type>::ExpandType
        };
        tokens.extend(out);
    }
}

impl ToTokens for KernelParam {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let name = &self.name;
        let ty = &self.normalized_ty;
        let span = self.span;
        tokens.extend(quote_spanned![span=>
            #name: #ty
        ]);
    }
}

impl Launch {
    fn kernel_phantom_data(&self) -> Option<TokenStream> {
        let generics = self.kernel_generics.clone();
        let declared_lifetimes = generics.declared_lifetimes();
        let declared_type_params = generics.declared_type_params();

        let used_lifetimes = self
            .comptime_params()
            .map(|param| &param.ty)
            .collect_lifetimes_cloned(&Purpose::Declare.into(), &declared_lifetimes);
        let used_type_params = self
            .comptime_params()
            .map(|param| &param.ty)
            .collect_type_params_cloned(&Purpose::Declare.into(), &declared_type_params);
        let lifetimes: Vec<_> = declared_lifetimes.difference(&used_lifetimes).collect();
        let type_params: Vec<_> = declared_type_params.difference(&used_type_params).collect();

        (!lifetimes.is_empty() && !type_params.is_empty())
            .then(|| quote![__ty: ::core::marker::PhantomData<(#(#lifetimes,)* #(#type_params),*)>])
    }

    fn define_body(&self) -> TokenStream {
        let io_map = self.io_mappings();
        let runtime_args = self.runtime_params().map(|it| &it.name);
        let comptime_args = self.comptime_params().map(|it| &it.name);

        quote! {
            #io_map
            __expand(&mut builder.context, #(#runtime_args.clone(),)* #(self.#comptime_args.clone()),*);
            builder.build(self.settings.clone())
        }
    }

    pub fn kernel_definition(&self) -> TokenStream {
        if self.args.is_launch() {
            let kernel = core_type("Kernel");
            let kernel_settings = prelude_type("KernelSettings");
            let kernel_definition: syn::Path = prelude_type("KernelDefinition");
            let kernel_id = core_type("KernelId");

            let kernel_name = self.kernel_name();
            let define = self.define_body();
            let kernel_doc = format!("{} Kernel", self.func.sig.name);

            let (generics, generic_names, where_clause) = self.kernel_generics.split_for_impl();
            let const_params = self.comptime_params();
            let phantom_data = self.kernel_phantom_data();
            let info = iter::once(format_ident!("settings"))
                .chain(self.comptime_params().map(|param| param.name.clone()));

            quote! {
                #[doc = #kernel_doc]
                #[derive(new)]
                pub struct #kernel_name #generics #where_clause {
                    settings: #kernel_settings,
                    #(#const_params,)*
                    #phantom_data
                }

                impl #generics #kernel for #kernel_name #generic_names #where_clause {
                    fn define(&self) -> #kernel_definition {
                        #define
                    }

                    fn id(&self) -> #kernel_id {
                        #kernel_id::new::<Self>().info((#(self.#info.clone()),*))
                    }
                }
            }
        } else {
            TokenStream::new()
        }
    }
}
