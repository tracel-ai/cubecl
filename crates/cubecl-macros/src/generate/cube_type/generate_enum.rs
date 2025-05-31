use crate::{
    parse::cube_type::{CubeTypeEnum, CubeTypeVariant, VariantKind},
    paths::prelude_type,
};
use proc_macro2::{Span, TokenStream};
use quote::{ToTokens, format_ident, quote};
use syn::{Ident, PathArguments, Type};

impl CubeTypeEnum {
    pub fn generate(&self, with_launch: bool) -> TokenStream {
        if with_launch {
            let args_ty = self.args_ty();
            let arg_settings_impl = self.arg_settings_impl();
            let launch_arg_impl = self.launch_arg_impl();

            let compilation_arg_ty = self.compilation_arg_ty();
            let launch_arg_expand_impl = self.launch_arg_expand_impl();

            quote! {
                #args_ty
                #arg_settings_impl
                #launch_arg_impl

                #compilation_arg_ty
                #launch_arg_expand_impl
            }
        } else {
            let expand_ty = self.expand_ty();
            let cube_type_impl = self.cube_type_impl();
            let expand_type_impl = self.expand_type_impl();

            quote! {
                #expand_ty
                #cube_type_impl
                #expand_type_impl
            }
        }
    }

    fn expand_ty(&self) -> proc_macro2::TokenStream {
        let name = &self.name_expand;
        let variants = self.variants.iter().map(CubeTypeVariant::expand_variant);
        let generics = &self.generics;
        let vis = &self.vis;

        quote! {
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
        let scope = prelude_type("Scope");
        let into_mut = prelude_type("IntoMut");
        let debug = prelude_type("CubeDebug");

        let name = &self.ident;
        let name_expand = &self.name_expand;
        let (generics, generic_names, where_clause) = self.generics.split_for_impl();

        let body_init = self.match_impl(
            quote! {self},
            self.variants
                .iter()
                .map(|v| v.map_body(name_expand, |f| quote!(#into_mut::into_mut(#f, scope))))
                .collect(),
        );

        let body_clone = self.match_impl(
            quote! {self},
            self.variants
                .iter()
                .map(|v| v.map_body(name_expand, |f| quote!(#f.clone())))
                .collect(),
        );

        let new_variant_functions = self
            .variants
            .iter()
            .map(|v| v.new_variant_function(name_expand, &generic_names));

        quote! {
            impl #generics #into_mut for #name_expand #generic_names #where_clause {
                fn into_mut(self, scope: &mut #scope) -> Self {
                    #body_init
                }
            }

            impl #generics #debug for #name #generic_names #where_clause {}

            impl #generics #debug for #name_expand #generic_names #where_clause {}

            impl #generics Clone for #name_expand #generic_names #where_clause {
                fn clone(&self) -> Self {
                    #body_clone
                }
            }

            #[allow(non_snake_case)]
            #[allow(unused)]
            impl #generics #name #generic_names #where_clause {
                #(
                    #new_variant_functions
                )*
            }

        }
    }

    fn args_ty(&self) -> proc_macro2::TokenStream {
        let name = Ident::new(&format!("{}Args", self.ident), Span::call_site());
        let vis = &self.vis;

        let generics = self.expanded_generics();

        let variants = self.variants.iter().map(|variant| {
            let variant_name = &variant.ident;
            match variant.kind {
                VariantKind::Named => {
                    let args = variant.fields.iter().map(|f| {
                        let field_name = &f.ident;
                        let field_ty = &f.ty;
                        quote! { #field_name: <#field_ty as LaunchArg>::RuntimeArg<'a, R> }
                    });
                    quote! {
                        #variant_name {
                            #(
                                #args,
                            )*
                        }
                    }
                }
                VariantKind::Unnamed => {
                    let args = variant.fields.iter().map(|f| {
                        let field_ty = &f.ty;
                        quote! { <#field_ty as LaunchArg>::RuntimeArg<'a, R> }
                    });
                    quote! {
                        #variant_name(#(#args),*)
                    }
                }
                VariantKind::Empty => {
                    quote! {
                        #variant_name
                    }
                }
            }
        });

        quote! {
            #vis enum #name #generics {
                #(
                    #variants
                ),*
            }
        }
    }

    fn arg_settings_impl(&self) -> proc_macro2::TokenStream {
        let arg_settings = prelude_type("ArgSettings");
        let kernel_launcher = prelude_type("KernelLauncher");
        let name = Ident::new(&format!("{}Args", self.ident), Span::call_site());

        let generics = self.expanded_generics();
        let (generics, generic_names, where_clause) = generics.split_for_impl();

        let body = self.match_impl(
            quote! {self},
            self.variants
                .iter()
                .map(|variant| {
                    let variant_name = &variant.ident;
                    match variant.kind {
                        VariantKind::Named => {
                            let args = &variant.field_names;
                            quote! {
                                #name::#variant_name { #(#args),* } => {
                                    #(
                                        #args.register(launcher);
                                    )*
                                }
                            }
                        }
                        VariantKind::Unnamed => {
                            let args = variant
                                .fields
                                .iter()
                                .enumerate()
                                .map(|(i, _)| Ident::new(&format!("_{i}"), Span::call_site()))
                                .collect::<Vec<_>>();
                            quote! {
                                #name::#variant_name(#(#args),*) => {
                                    #(
                                        #args.register(launcher);
                                    )*
                                }
                            }
                        }
                        VariantKind::Empty => {
                            quote! {
                                #name::#variant_name => {}
                            }
                        }
                    }
                })
                .collect(),
        );

        quote! {
            impl #generics #arg_settings<R> for #name #generic_names #where_clause {
                fn register(&self, launcher: &mut #kernel_launcher<R>) {
                    #body
                }
            }
        }
    }

    fn launch_arg_impl(&self) -> proc_macro2::TokenStream {
        let launch_arg = prelude_type("LaunchArg");

        let name = &self.ident;
        let name_args = Ident::new(&format!("{}Args", self.ident), Span::call_site());
        let compilation_arg =
            Ident::new(&format!("{}CompilationArg", self.ident), Span::call_site());

        let (generics, generic_names, where_clause) = self.generics.split_for_impl();

        let assoc_generics = self.assoc_generics();
        let all = self.expanded_generics();
        let (_, all_generic_names, _) = all.split_for_impl();

        let branches = self
            .variants
            .iter()
            .map(|variant| {
                let variant_name = &variant.ident;
                match variant.kind {
                    VariantKind::Named => {
                        let args = &variant.field_names;
                        let body = variant.fields.iter().map(|f| {
                            let ty = add_double_colon_in_type(f.ty.clone());
                            let name = &f.ident;
                            quote! { #name: #ty::compilation_arg::<R>(#name), }
                        });
                        quote! {
                            #name_args::#variant_name { #(#args),* } => #compilation_arg::#variant_name {
                                #(
                                    #body
                                )*
                            }
                        }
                    }
                    VariantKind::Unnamed => {
                        let args = variant
                            .fields
                            .iter()
                            .enumerate()
                            .map(|(i, _)| Ident::new(&format!("_{i}"), Span::call_site()));
                        let body = variant.fields.iter().enumerate().map(|(i, f)| {
                            let ty = add_double_colon_in_type(f.ty.clone());
                            let name = Ident::new(&format!("_{i}"), Span::call_site());
                            quote! { #ty::compilation_arg::<R>(#name), }
                        });
                        quote! {
                            #name_args::#variant_name(#(#args),*) => #compilation_arg::#variant_name (
                                #(
                                    #body
                                )*
                            )
                        }
                    }
                    VariantKind::Empty => {
                        quote! {
                            #name_args::#variant_name => #compilation_arg::#variant_name
                        }
                    }
                }
            })
            .collect();

        let body = self.match_impl(quote! {runtime_arg}, branches);

        quote! {
            impl #generics #launch_arg for #name #generic_names #where_clause {
                type RuntimeArg #assoc_generics = #name_args #all_generic_names;

                fn compilation_arg #assoc_generics(runtime_arg: &Self::RuntimeArg<'a, R>) -> Self::CompilationArg {
                    #body
                }
            }
        }
    }

    fn compilation_arg_ty(&self) -> proc_macro2::TokenStream {
        let compilation_arg = prelude_type("CompilationArg");

        let name = Ident::new(&format!("{}CompilationArg", self.ident), Span::call_site());
        let vis = &self.vis;

        let generics = &self.generics;

        let variants = self.variants.iter().map(|variant| {
            let variant_name = &variant.ident;
            match variant.kind {
                VariantKind::Named => {
                    let args = variant.fields.iter().map(|f| {
                        let field_name = &f.ident;
                        let field_ty = &f.ty;
                        quote! { #field_name: <#field_ty as LaunchArgExpand>::CompilationArg }
                    });
                    quote! {
                        #variant_name {
                            #(
                                #args,
                            )*
                        }
                    }
                }
                VariantKind::Unnamed => {
                    let args = variant.fields.iter().map(|f| {
                        let field_ty = &f.ty;
                        quote! { <#field_ty as LaunchArgExpand>::CompilationArg }
                    });
                    quote! {
                        #variant_name(#(#args),*)
                    }
                }
                VariantKind::Empty => {
                    quote! {
                        #variant_name
                    }
                }
            }
        });

        let (generics_impl, generic_names, where_clause) = self.generics.split_for_impl();

        let body_clone = self.match_impl(
            quote! {self},
            self.variants
                .iter()
                .map(|v| v.map_body(&name, |f| quote!(#f.clone())))
                .collect(),
        );

        let body_partial_eq = self.match_impl(
            quote! {(self, other)},
            self.variants
                .iter()
                .map(|v| v.partial_eq_body(&name))
                .chain(std::iter::once(quote! { _ => false}))
                .collect(),
        );

        let body_hash = self.match_impl(
            quote! {self},
            self.variants
                .iter()
                .map(|v| v.for_each_body(&name, |f| quote!(#f.hash(state))))
                .collect(),
        );

        let body_debug = self.match_impl(
            quote! {self},
            self.variants.iter().map(|v| v.debug_body(&name)).collect(),
        );

        quote! {
            #[derive(serde::Serialize, serde::Deserialize)]
            #[serde(bound(serialize = "", deserialize = ""))]
            #vis enum #name #generics {
                #(
                    #variants
                ),*
            }

            impl #generics_impl Clone for #name #generic_names #where_clause {
                fn clone(&self) -> Self {
                    #body_clone
                }
            }

            impl #generics_impl PartialEq for #name #generic_names #where_clause {
                fn eq(&self, other: &Self) -> bool {
                    #body_partial_eq
                }
            }

            impl #generics_impl Eq for #name #generic_names #where_clause {}

            impl #generics_impl core::hash::Hash for #name #generic_names #where_clause {
                fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
                    #body_hash;
                }
            }


            impl #generics_impl core::fmt::Debug for #name #generic_names #where_clause {
                fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                    #body_debug
                }
            }


            impl #generics_impl #compilation_arg for #name #generic_names #where_clause {}
        }
    }

    fn launch_arg_expand_impl(&self) -> proc_macro2::TokenStream {
        let launch_arg_expand = prelude_type("LaunchArgExpand");
        let cube_type = prelude_type("CubeType");
        let kernel_builder = prelude_type("KernelBuilder");

        let name = &self.ident;
        let name_expand = Ident::new(&format!("{}Expand", self.ident), Span::call_site());
        let compilation_arg =
            Ident::new(&format!("{}CompilationArg", self.ident), Span::call_site());

        let (generics, generic_names, where_clause) = self.generics.split_for_impl();

        let branches_expand = self
            .variants
            .iter()
            .map(|variant| {
                let variant_name = &variant.ident;
                match variant.kind {
                    VariantKind::Named => {
                        let args = &variant.field_names;
                        let body = variant.fields.iter().map(|f| {
                            let ty = add_double_colon_in_type(f.ty.clone());
                            let name = &f.ident;
                            quote! { #name: #ty::expand(#name, builder), }
                        });
                        quote! {
                            #compilation_arg::#variant_name { #(#args),* } => #name_expand::#variant_name {
                                #(
                                    #body
                                )*
                            }
                        }
                    }
                    VariantKind::Unnamed => {
                        let args = variant
                            .fields
                            .iter()
                            .enumerate()
                            .map(|(i, _)| Ident::new(&format!("_{i}"), Span::call_site()));
                        let compilation_args = variant.fields.iter().enumerate().map(|(i, f)| {
                            let ty = add_double_colon_in_type(f.ty.clone());
                            let name = Ident::new(&format!("_{i}"), Span::call_site());
                            quote! { #ty::expand(#name, builder), }
                        });
                        quote! {
                            #compilation_arg::#variant_name(#(#args),*) => #name_expand::#variant_name (
                                #(
                                    #compilation_args
                                )*
                            )
                        }
                    }
                    VariantKind::Empty => {
                        quote! {
                            #compilation_arg::#variant_name => #name_expand::#variant_name
                        }
                    }
                }
            })
            .collect();

        let body_expand = self.match_impl(quote! {arg}, branches_expand);

        let branches_expand_output = self
            .variants
            .iter()
            .map(|variant| {
                let variant_name = &variant.ident;
                match variant.kind {
                    VariantKind::Named => {
                        let args = &variant.field_names;
                        let body = variant.fields.iter().map(|f| {
                            let ty = add_double_colon_in_type(f.ty.clone());
                            let name = &f.ident;
                            quote! { #name: #ty::expand_output(#name, builder), }
                        });
                        quote! {
                            #compilation_arg::#variant_name { #(#args),* } => #name_expand::#variant_name {
                                #(
                                    #body
                                )*
                            }
                        }
                    }
                    VariantKind::Unnamed => {
                        let args = variant
                            .fields
                            .iter()
                            .enumerate()
                            .map(|(i, _)| Ident::new(&format!("_{i}"), Span::call_site()));
                        let compilation_args = variant.fields.iter().enumerate().map(|(i, f)| {
                            let ty = add_double_colon_in_type(f.ty.clone());
                            let name = Ident::new(&format!("_{i}"), Span::call_site());
                            quote! { #ty::expand_output(#name, builder), }
                        });
                        quote! {
                            #compilation_arg::#variant_name(#(#args),*) => #name_expand::#variant_name (
                                #(
                                    #compilation_args
                                )*
                            )
                        }
                    }
                    VariantKind::Empty => {
                        quote! {
                            #compilation_arg::#variant_name => #name_expand::#variant_name
                        }
                    }
                }
            })
            .collect();

        let body_expand_output = self.match_impl(quote! {arg}, branches_expand_output);

        quote! {
            impl #generics #launch_arg_expand for #name #generic_names #where_clause {
                type CompilationArg = #compilation_arg #generic_names;

                fn expand(arg: &Self::CompilationArg, builder: &mut #kernel_builder) -> <Self as #cube_type>::ExpandType {
                    #body_expand
                }

                fn expand_output(
                    arg: &Self::CompilationArg,
                    builder: &mut #kernel_builder,
                ) -> <Self as #cube_type>::ExpandType {
                    #body_expand_output
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

        let fields = self.fields.iter().map(|field| {
            let ty = add_double_colon_in_type(field.ty.clone());
            match &field.ident {
                Some(name) => {
                    quote! {#name: <#ty as #cube_type>::ExpandType}
                }
                None => quote! {<#ty as #cube_type>::ExpandType},
            }
        });

        match self.kind {
            VariantKind::Named => quote![#name { #(#fields),* } ],
            VariantKind::Unnamed => quote![#name ( #(#fields),* ) ],
            VariantKind::Empty => quote!( #name ),
        }
    }

    fn map_body<F: Fn(&Ident) -> TokenStream>(
        &self,
        ident_ty_expand: &Ident,
        fn_call: F,
    ) -> TokenStream {
        let name = &self.ident;
        let body = self.field_names.iter().map(|name| {
            let called = fn_call(name);
            if let VariantKind::Named = self.kind {
                quote! {
                    #name: #called
                }
            } else {
                quote! {
                    #called
                }
            }
        });

        let body = match self.kind {
            VariantKind::Named => quote![#ident_ty_expand::#name { #(#body),* } ],
            VariantKind::Unnamed => quote![#ident_ty_expand::#name ( #(#body),* ) ],
            VariantKind::Empty => quote![#ident_ty_expand::#name],
        };

        self.run_on_variants(ident_ty_expand, body)
    }

    fn for_each_body<F: Fn(&Ident) -> TokenStream>(
        &self,
        ident_ty_expand: &Ident,
        fn_call: F,
    ) -> TokenStream {
        let body = self.field_names.iter().map(|name| {
            let called = fn_call(name);
            quote! {
                #called
            }
        });

        let body = match self.kind {
            VariantKind::Named | VariantKind::Unnamed => quote![ {#(#body;)*} ],
            VariantKind::Empty => quote![{}],
        };

        self.run_on_variants(ident_ty_expand, body)
    }

    fn partial_eq_body(&self, ident_ty_expand: &Ident) -> TokenStream {
        let body = self.field_names.iter().map(|name| {
            let ident_0 = format_ident!("{name}_0");
            let ident_1 = format_ident!("{name}_1");
            quote! { #ident_0 == #ident_1 }
        });

        let body = match self.kind {
            VariantKind::Named => quote![{ #(#body)&& * } ],
            VariantKind::Unnamed => quote![{ #(#body)&& * } ],
            VariantKind::Empty => quote![{ true }],
        };

        self.run_on_variant_pairs(ident_ty_expand, body)
    }

    fn debug_body(&self, ident_ty_expand: &Ident) -> TokenStream {
        let body = self.field_names.iter().map(|name| match self.kind {
            VariantKind::Named => {
                let field = name.to_string();
                quote! { .field(#field, #name) }
            }
            VariantKind::Unnamed => quote! { .field(#name) },
            VariantKind::Empty => quote! {},
        });

        let variant = &self.ident;
        let name = quote!(#ident_ty_expand::#variant).to_string();

        let body = match self.kind {
            VariantKind::Named => quote![
                f.debug_struct(#name)
                #(#body)*
                .finish()
            ],
            VariantKind::Unnamed => quote![
                f.debug_tuple(#name)
                #(#body)*
                .finish()
            ],
            VariantKind::Empty => quote![write!(f, #name)],
        };

        self.run_on_variants(ident_ty_expand, body)
    }

    fn new_variant_function(
        &self,
        ident_ty_expand: &Ident,
        generics: &syn::TypeGenerics,
    ) -> TokenStream {
        let cube_type = prelude_type("CubeType");
        let scope = prelude_type("Scope");
        let ident = &self.ident;
        let base_function = Ident::new(&format!("new_{ident}"), ident.span());
        let expand_function = Ident::new(&format!("__expand_new_{ident}"), ident.span());

        let turbofish = generics.as_turbofish();

        match self.kind {
            VariantKind::Named => {
                let args_with_base_types = self.fields.iter().map(|f| {
                    let ty = &f.ty;
                    let name = &f.ident;
                    quote! {
                        #name: #ty
                    }
                });

                let args_with_expand_types = self.fields.iter().map(|f| {
                    let ty = &f.ty;
                    let name = &f.ident;
                    quote! {
                        #name: <#ty as #cube_type>::ExpandType
                    }
                });

                let args = self.fields.iter().map(|f| &f.ident);

                quote! {
                    pub fn #base_function(#(#args_with_base_types),*) -> Self {
                        cubecl::unexpanded!()
                    }

                    pub fn #expand_function(_: &mut #scope, #(#args_with_expand_types),*) -> #ident_ty_expand #generics {
                        #ident_ty_expand #turbofish ::#ident {#(#args),*}
                    }
                }
            }
            VariantKind::Unnamed => {
                let args_with_base_types = self.fields.iter().enumerate().map(|(i, f)| {
                    let ty = &f.ty;
                    let name = Ident::new(&format!("_{i}"), Span::call_site());
                    quote! {
                        #name: #ty
                    }
                });

                let args_with_expand_types = self.fields.iter().enumerate().map(|(i, f)| {
                    let ty = &f.ty;
                    let name = Ident::new(&format!("_{i}"), Span::call_site());
                    quote! {
                        #name: <#ty as #cube_type>::ExpandType
                    }
                });

                let args = self
                    .fields
                    .iter()
                    .enumerate()
                    .map(|(i, _)| Ident::new(&format!("_{i}"), Span::call_site()));

                quote! {
                    pub fn #base_function(#(#args_with_base_types),*) -> Self {
                        cubecl::unexpanded!()
                    }

                    pub fn #expand_function(_: &mut #scope, #(#args_with_expand_types),*) -> #ident_ty_expand #generics {
                        #ident_ty_expand #turbofish ::#ident(#(#args),*)
                    }
                }
            }
            VariantKind::Empty => {
                quote! {
                    pub fn #base_function() -> Self {
                        cubecl::unexpanded!()
                    }

                    pub fn #expand_function(_: &mut #scope) -> #ident_ty_expand #generics {
                        #ident_ty_expand #turbofish ::#ident
                    }
                }
            }
        }
    }

    fn run_on_variants(&self, parent_ty: &Ident, body: TokenStream) -> TokenStream {
        let ident = &self.ident;
        let decl = &self.field_names;

        match self.kind {
            VariantKind::Named => quote! {
                #parent_ty::#ident {
                    #(#decl),*
                } => #body
            },
            VariantKind::Unnamed => quote! (
                #parent_ty::#ident (
                    #(#decl),*
                ) => #body
            ),
            VariantKind::Empty => quote! {
                #parent_ty::#ident => #body
            },
        }
    }

    fn run_on_variant_pairs(&self, parent_ty: &Ident, body: TokenStream) -> TokenStream {
        let ident = &self.ident;

        match self.kind {
            VariantKind::Named => {
                let decl_0 = self.field_names.iter().map(|name| {
                    let binding = Ident::new(&format!("{name}_0"), Span::call_site());
                    quote! { #name: #binding}
                });
                let decl_1 = self.field_names.iter().map(|name| {
                    let binding = Ident::new(&format!("{name}_1"), Span::call_site());
                    quote! { #name: #binding}
                });
                quote! {
                    (
                        #parent_ty::#ident {
                            #(#decl_0),*
                        },
                        #parent_ty::#ident {
                            #(#decl_1),*
                        }
                    ) => #body
                }
            }
            VariantKind::Unnamed => {
                let decl_0 = self
                    .field_names
                    .iter()
                    .map(|name| Ident::new(&format!("{name}_0"), Span::call_site()));
                let decl_1 = self
                    .field_names
                    .iter()
                    .map(|name| Ident::new(&format!("{name}_1"), Span::call_site()));
                quote! (
                    (
                        #parent_ty::#ident (
                            #(#decl_0),*
                        ),
                        #parent_ty::#ident (
                            #(#decl_1),*
                        )
                    ) => #body
                )
            }
            VariantKind::Empty => quote! {
                (#parent_ty::#ident, #parent_ty::#ident) => #body
            },
        }
    }
}

fn add_double_colon_in_type(mut ty: Type) -> TokenStream {
    match ty {
        Type::Path(ref mut ty) => ty
            .path
            .segments
            .last_mut()
            .map(|ty| match ty.clone().arguments {
                PathArguments::AngleBracketed(arg) => {
                    let mut new_ty = ty.clone();
                    new_ty.arguments = PathArguments::None;
                    quote! { #new_ty :: #arg }
                }
                _ => ty.to_token_stream(),
            })
            .expect("type path must contains at least a segment"),
        ty => ty.to_token_stream(),
    }
}
