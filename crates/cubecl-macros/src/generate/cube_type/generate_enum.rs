use crate::{
    parse::cube_type::{CubeTypeEnum, CubeTypeVariant, VariantKind},
    paths::prelude_type,
};
use proc_macro2::{Span, TokenStream};
use quote::{quote, ToTokens};
use syn::{Ident, PathArguments, Type};

impl CubeTypeEnum {
    pub fn generate(&self, with_launch: bool) -> TokenStream {
        let expand_ty = self.expand_ty();
        let cube_type_impl = self.cube_type_impl();
        let expand_type_impl = self.expand_type_impl();

        if with_launch {
            let args_ty = self.args_ty();
            let arg_settings_impl = self.arg_settings_impl();
            let launch_arg_impl = self.launch_arg_impl();

            let compilation_arg_ty = self.compilation_arg_ty();
            let launch_arg_expand_impl = self.launch_arg_expand_impl();

            quote! {
                #expand_ty
                #cube_type_impl
                #expand_type_impl

                #args_ty
                #arg_settings_impl
                #launch_arg_impl

                #compilation_arg_ty
                #launch_arg_expand_impl
            }
        } else {
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
        let context = prelude_type("Scope");
        let into_runtime = prelude_type("IntoRuntime");
        let init = prelude_type("Init");
        let debug = prelude_type("CubeDebug");

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
                .map(|v| v.runtime_body(name, name_expand))
                .collect(),
        );

        let new_variant_functions = self
            .variants
            .iter()
            .map(|v| v.new_variant_function(name_expand, &generic_names));

        quote! {
            impl #generics #init for #name_expand #generic_names #where_clause {
                fn init(self, context: &mut #context) -> Self {
                    #body_init
                }
            }

            impl #generics #debug for #name #generic_names #where_clause {}

            impl #generics #debug for #name_expand #generic_names #where_clause {}

            impl #generics #into_runtime for #name #generic_names #where_clause {
                fn __expand_runtime_method(self, context: &mut #context) -> Self::ExpandType {
                    let expand = #body_into_runtime;
                    #init::init(expand, context)
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

        let branches = self
            .variants
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
            .collect();

        let body = self.match_impl(quote! {self}, branches);

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

        quote! {
            #[derive(Clone, serde::Serialize, serde::Deserialize, Hash, PartialEq, Eq, Debug)]
            #[serde(bound(serialize = "", deserialize = ""))]
            #vis enum #name #generics {
                #(
                    #variants
                ),*
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

    fn runtime_body(&self, ident_ty: &Ident, ident_ty_expand: &Ident) -> TokenStream {
        let name = &self.ident;
        let into_runtime = prelude_type("IntoRuntime");
        let body = self.field_names.iter().map(|name| {
            if let VariantKind::Named = self.kind {
                quote! {
                    #name: #into_runtime::__expand_runtime_method(#name, context)
                }
            } else {
                quote! {
                    #into_runtime::__expand_runtime_method(#name, context)
                }
            }
        });

        let body = match self.kind {
            VariantKind::Named => quote![#ident_ty_expand::#name { #(#body),*} ],
            VariantKind::Unnamed => quote![#ident_ty_expand::#name ( #(#body),* ) ],
            VariantKind::Empty => quote![#ident_ty_expand::#name],
        };

        self.run_on_variants(ident_ty, body)
    }

    fn init_body(&self, ident_ty_expand: &Ident) -> TokenStream {
        let name = &self.ident;
        let init = prelude_type("Init");
        let body = self.field_names.iter().map(|name| {
            if let VariantKind::Named = self.kind {
                quote! {
                    #name: #init::init(#name, context)
                }
            } else {
                quote! {
                    #init::init(#name, context)
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

    fn new_variant_function(
        &self,
        ident_ty_expand: &Ident,
        generics: &syn::TypeGenerics,
    ) -> TokenStream {
        let cube_type = prelude_type("CubeType");
        let context = prelude_type("Scope");
        let ident = &self.ident;
        let base_function = Ident::new(&format!("new_{}", ident), ident.span());
        let expand_function = Ident::new(&format!("__expand_new_{}", ident), ident.span());

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

                    pub fn #expand_function(_: &mut #context, #(#args_with_expand_types),*) -> #ident_ty_expand #generics {
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

                    pub fn #expand_function(_: &mut #context, #(#args_with_expand_types),*) -> #ident_ty_expand #generics {
                        #ident_ty_expand #turbofish ::#ident(#(#args),*)
                    }
                }
            }
            VariantKind::Empty => {
                quote! {
                    pub fn #base_function() -> Self {
                        cubecl::unexpanded!()
                    }

                    pub fn #expand_function(_: &mut #context) -> #ident_ty_expand #generics {
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
