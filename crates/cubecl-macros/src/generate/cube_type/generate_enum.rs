use crate::{
    parse::cube_type::{CubeTypeEnum, CubeTypeVariant},
    paths::prelude_type,
};
use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

impl CubeTypeEnum {
    pub fn generate(&self, with_launch: bool) -> TokenStream {
        assert!(with_launch == false, "Can't create launchable enum yet.");

        let expand_ty = self.expand_ty();
        let cube_type_impl = self.cube_type_impl();
        let expand_type_impl = self.expand_type_impl();

        quote! {
           #expand_ty
           #cube_type_impl
           #expand_type_impl
        }
    }

    fn expand_ty(&self) -> proc_macro2::TokenStream {
        let name = &self.name_expand;
        let variants = self.variants.iter().map(CubeTypeVariant::expand_variant);
        let generics = &self.generics;
        let vis = &self.vis;

        quote! {
            #[derive(Clone)]
            #vis enum #name #generics {
                #(#variants),*
            }
        }
    }
    fn cube_type_impl(&self) -> proc_macro2::TokenStream {
        let cube_type = prelude_type("CubeType");
        let name = &self.ident;
        let name_expand = &self.name_expand;

        let (generics, generic_names, where_clause) = self.generics.split_for_impl();

        quote! {
            impl #generics #cube_type for #name #generic_names #where_clause {
                type ExpandType = #name_expand #generic_names;
            }
        }
    }

    fn expand_type_impl(&self) -> proc_macro2::TokenStream {
        let context = prelude_type("CubeContext");
        let into_runtime = prelude_type("IntoRuntime");
        let init = prelude_type("Init");

        let name = &self.ident;
        let name_expand = &self.name_expand;
        let (generics, generic_names, where_clause) = self.generics.split_for_impl();

        let body_init = self.match_impl(
            quote! {self},
            self.variants
                .iter()
                .map(|v| v.init_body(name_expand))
                .collect(),
        );

        let body_into_runtime = self.match_impl(
            quote! {self},
            self.variants
                .iter()
                .map(|v| v.into_runtime_body(name, name_expand))
                .collect(),
        );

        quote! {
            impl #generics #init for #name_expand #generic_names #where_clause {
                fn init(self, context: &mut #context) -> Self {
                    #body_init
                }
            }

            impl #generics #into_runtime for #name #generic_names #where_clause {
                fn __expand_runtime_method(self, context: &mut CubeContext) -> Self::ExpandType {
                    let expand = #body_into_runtime;
                    #init::init(expand, context)
                }
            }
        }
    }

    fn match_impl(
        &self,
        match_input_tokens: TokenStream,
        branches: Vec<TokenStream>,
    ) -> TokenStream {
        quote! {
            match #match_input_tokens {
                #(#branches,)*
            }
        }
    }
}

impl CubeTypeVariant {
    fn expand_variant(&self) -> TokenStream {
        let name = &self.ident;
        let cube_type = prelude_type("CubeType");
        let mut named = false;

        let fields = self
            .fields
            .iter()
            .map(|field| {
                let ty = &field.ty;
                match &field.ident {
                    Some(name) => {
                        named = true;
                        quote! {#name: <#ty as #cube_type>::ExpandType}
                    }
                    None => quote! {<#ty as #cube_type>::ExpandType},
                }
            })
            .collect::<Vec<_>>();

        if named {
            quote![#name { #(#fields),* } ]
        } else {
            quote![#name ( #(#fields),* ) ]
        }
    }

    fn into_runtime_body(&self, ident_ty: &Ident, ident_ty_expand: &Ident) -> TokenStream {
        let name = &self.ident;
        let into_runtime = prelude_type("IntoRuntime");
        let (named, names) = self.fields_names();
        let body = names.iter().map(|name| {
            if named {
                quote! {
                    #name: #into_runtime::__expand_runtime_method(#name, context)
                }
            } else {
                quote! {
                    #into_runtime::__expand_runtime_method(#name, context)
                }
            }
        });

        let body = if named {
            quote![#ident_ty_expand::#name { #(#body),*} ]
        } else {
            quote![#ident_ty_expand::#name ( #(#body),* ) ]
        };

        self.run_on_variants(named, ident_ty, &names, body)
    }

    fn init_body(&self, ident_ty_expand: &Ident) -> TokenStream {
        let name = &self.ident;
        let init = prelude_type("Init");
        let (named, names) = self.fields_names();
        let body = names.iter().map(|name| {
            if named {
                quote! {
                    #name: #init::init(#name, context)
                }
            } else {
                quote! {
                    #init::init(#name, context)
                }
            }
        });

        let body = if named {
            quote![#ident_ty_expand::#name { #(#body),*} ]
        } else {
            quote![#ident_ty_expand::#name ( #(#body),* ) ]
        };

        self.run_on_variants(named, ident_ty_expand, &names, body)
    }

    fn fields_names(&self) -> (bool, Vec<Ident>) {
        let mut named = false;

        let names = self
            .fields
            .iter()
            .enumerate()
            .map(|(i, field)| match &field.ident {
                Some(name) => {
                    named = true;
                    name.clone()
                }
                None => Ident::new(format!("arg_{i}").as_str(), self.ident.span()),
            })
            .collect::<Vec<_>>();

        (named, names)
    }

    fn run_on_variants(
        &self,
        named: bool,
        parent_ty: &Ident,
        decl: &[Ident],
        body: TokenStream,
    ) -> TokenStream {
        let ident = &self.ident;

        if named {
            quote! {
                #parent_ty::#ident {
                    #(#decl),*
                } => #body
            }
        } else {
            quote! (
                #parent_ty::#ident (
                    #(#decl),*
                ) => #body
            )
        }
    }
}
