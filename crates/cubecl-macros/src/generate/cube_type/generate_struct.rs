use proc_macro2::TokenStream;
use quote::quote;
use syn::{Ident, Type, Visibility};

use crate::{
    parse::cube_type::{CubeTypeStruct, TypeField},
    paths::prelude_type,
};

impl CubeTypeStruct {
    pub fn generate(&self, with_launch: bool) -> TokenStream {
        if with_launch {
            let launch_ty = self.launch_ty();
            let launch_new = self.launch_new();
            let arg_settings_impl = self.arg_settings_impl();
            let launch_arg_impl = self.launch_arg_impl();

            quote! {
                #launch_ty
                #launch_new
                #arg_settings_impl
                #launch_arg_impl
            }
        } else {
            let expand_ty = self.expand_ty();
            let clone_expand = self.clone_expand();
            let cube_type_impl = self.cube_type_impl();
            let expand_type_impl = self.expand_type_impl();

            quote! {
                #expand_ty
                #clone_expand
                #cube_type_impl
                #expand_type_impl
            }
        }
    }

    fn expand_ty(&self) -> proc_macro2::TokenStream {
        let fields = self.fields.iter().map(TypeField::expand_field);
        let name = &self.name_expand;
        let generics = &self.generics;
        let vis = &self.vis;

        quote! {
            #vis struct #name #generics {
                #(#fields),*
            }
        }
    }

    fn clone_expand(&self) -> proc_macro2::TokenStream {
        let fields = self.fields.iter().map(TypeField::clone_field);
        let name = &self.name_expand;
        let generics = &self.generics;

        let (generics_impl, generics_use, where_clause) = generics.split_for_impl();

        quote! {
            impl #generics_impl Clone for #name #generics_use #where_clause {
                fn clone(&self) -> Self {
                    Self {
                        #(#fields),*
                    }
                }
            }
        }
    }

    fn launch_ty(&self) -> proc_macro2::TokenStream {
        let name = &self.name_launch;
        let fields = self.fields.iter().map(TypeField::launch_field);
        let generics = self.expanded_generics();
        let vis = &self.vis;

        quote! {
            #vis struct #name #generics {
                _phantom_runtime: core::marker::PhantomData<R>,
                _phantom_a: core::marker::PhantomData<&'a ()>,
                #(#fields),*
            }
        }
    }

    fn launch_new(&self) -> proc_macro2::TokenStream {
        let args = self.fields.iter().map(TypeField::launch_new_arg);
        let fields = self.fields.iter().map(|field| &field.ident);
        let name = &self.name_launch;

        let generics = self.expanded_generics();
        let (generics_impl, generics_use, where_clause) = generics.split_for_impl();
        let vis = &self.vis;

        quote! {
            impl #generics_impl #name #generics_use #where_clause {
                /// New kernel
                #[allow(clippy::too_many_arguments)]
                #vis fn new(#(#args),*) -> Self {
                    Self {
                        _phantom_runtime: core::marker::PhantomData,
                        _phantom_a: core::marker::PhantomData,
                        #(#fields),*
                    }
                }
            }
        }
    }

    fn arg_settings_impl(&self) -> proc_macro2::TokenStream {
        let arg_settings = prelude_type("ArgSettings");
        let kernel_launcher = prelude_type("KernelLauncher");
        let name = &self.name_launch;
        let register_body = self
            .fields
            .iter()
            .filter(|f| !f.comptime.is_present())
            .map(TypeField::split)
            .map(|(_, ident, _, _)| quote![self.#ident.register(launcher)]);

        let generics = self.expanded_generics();
        let (generics, generic_names, where_clause) = generics.split_for_impl();

        quote! {
            impl #generics #arg_settings<R> for #name #generic_names #where_clause {
                fn register(&self, launcher: &mut #kernel_launcher<R>) {
                    #(#register_body;)*
                }
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

    fn compilation_ty_ident(&self) -> Ident {
        Ident::new(
            format!("{}CompilationArg", self.ident).as_str(),
            self.ident.span(),
        )
    }

    fn compilation_arg_impl(&self, name: &Ident) -> TokenStream {
        let launch_arg = prelude_type("LaunchArg");
        let fields = self.fields.iter().map(|field| {
            let name = &field.ident;
            let ty = &field.ty;

            if field.comptime.is_present() {
                quote! {
                   #name: runtime_arg.#name.clone()
                }
            } else {
                quote! {
                   #name: <#ty as #launch_arg>::compilation_arg::<R>(&runtime_arg.#name)
                }
            }
        });
        quote! {
            #name {
                #(#fields,)*
            }
        }
    }

    fn compilation_ty(&self, name: &Ident) -> proc_macro2::TokenStream {
        let name_debug = &self.ident;
        let fields = self.fields.iter().map(TypeField::compilation_arg_field);
        let generics = &self.generics;
        let (type_generics_names, impl_generics, where_generics) = self.generics.split_for_impl();
        let vis = &self.vis;

        fn generate<'a, F: Fn(&Ident) -> TokenStream>(
            fields: impl Iterator<Item = &'a TypeField>,
            func: F,
        ) -> Vec<TokenStream> {
            fields
                .map(|field| func(field.ident.as_ref().unwrap()))
                .collect::<Vec<_>>()
        }

        let clone = generate(self.fields.iter(), |name| quote!(#name: self.#name.clone()));
        let hash = generate(self.fields.iter(), |name| quote!(self.#name.hash(state)));
        let partial_eq = generate(
            self.fields.iter(),
            |name| quote!(self.#name.eq(&other.#name)),
        );
        let debug = generate(self.fields.iter(), |name| {
            quote!(f.write_fmt(format_args!("{}: {:?},", stringify!(#name), &self.#name))?)
        });

        quote! {
            #vis struct #name #generics {
                #(#fields),*
            }

            impl #type_generics_names Clone for #name #impl_generics #where_generics {
                fn clone(&self) -> Self {
                    Self {
                        #(#clone,)*
                    }
                }
            }

            impl #type_generics_names CompilationArg for #name #impl_generics #where_generics { }

            impl #type_generics_names core::hash::Hash for #name #impl_generics #where_generics {
                fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                    #(#hash;)*
                }
            }

            impl #type_generics_names core::cmp::PartialEq for #name #impl_generics #where_generics {
                fn eq(&self, other: &Self) -> bool {
                    #(#partial_eq &&)* true
                }
            }

            impl #type_generics_names core::fmt::Debug for #name #impl_generics #where_generics {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    f.write_str(stringify!(#name_debug))?;
                    f.write_str("{")?;
                    #(#debug;)*
                    f.write_str("}")?;

                    Ok(())
                }
            }
            impl #type_generics_names core::cmp::Eq for #name #impl_generics #where_generics { }
        }
    }

    fn launch_arg_impl(&self) -> proc_macro2::TokenStream {
        let launch_arg = prelude_type("LaunchArg");
        let body_input =
            self.fields
                .iter()
                .map(TypeField::split)
                .map(|(_vis, name, ty, is_comptime)| {
                    if is_comptime {
                        quote![#name: arg.#name.clone()]
                    } else {
                        quote![#name: <#ty as #launch_arg>::expand(&arg.#name, builder)]
                    }
                });
        let body_output =
            self.fields
                .iter()
                .map(TypeField::split)
                .map(|(_vis, name, ty, is_comptime)| {
                    if is_comptime {
                        quote![#name: arg.#name.clone()]
                    } else {
                        quote![#name: <#ty as #launch_arg>::expand_output(&arg.#name, builder)]
                    }
                });

        let name = &self.ident;
        let name_launch = &self.name_launch;
        let name_expand = &self.name_expand;

        let (type_generics, type_generic_names, where_clause) = self.generics.split_for_impl();

        let (_, compilation_generics, _) = self.generics.split_for_impl();
        let assoc_generics = self.assoc_generics();
        let all = self.expanded_generics();
        let (_, all_generic_names, _) = all.split_for_impl();

        let compilation_ident = self.compilation_ty_ident();
        let compilation_arg = self.compilation_ty(&compilation_ident);
        let compilation_arg_impl = self.compilation_arg_impl(&compilation_ident);

        quote! {
            #compilation_arg

            impl #type_generics #launch_arg for #name #type_generic_names #where_clause {
                type RuntimeArg #assoc_generics = #name_launch #all_generic_names;
                type CompilationArg = #compilation_ident #compilation_generics;

                fn compilation_arg<'a, R: Runtime>(
                    runtime_arg: &Self::RuntimeArg<'a, R>,
                ) -> Self::CompilationArg {
                    #compilation_arg_impl
                }

                fn expand(
                    arg: &Self::CompilationArg,
                    builder: &mut KernelBuilder,
                ) -> <Self as CubeType>::ExpandType {
                    #name_expand {
                        #(#body_input),*
                    }
                }
                fn expand_output(
                    arg: &Self::CompilationArg,
                    builder: &mut KernelBuilder,
                ) -> <Self as CubeType>::ExpandType {
                    #name_expand {
                        #(#body_output),*
                    }
                }
            }
        }
    }

    fn expand_type_impl(&self) -> proc_macro2::TokenStream {
        let into_mut = prelude_type("IntoMut");
        let debug = prelude_type("CubeDebug");
        let scope = prelude_type("Scope");
        let name_expand = &self.name_expand;
        let (generics, generic_names, where_clause) = self.generics.split_for_impl();
        let body = self
            .fields
            .iter()
            .map(TypeField::split)
            .map(|(_, ident, _, is_comptime)| {
                if is_comptime {
                    quote![#ident: self.#ident]
                } else {
                    quote![#ident: #into_mut::into_mut(self.#ident, scope)]
                }
            });

        quote! {
            impl #generics #into_mut for #name_expand #generic_names #where_clause {
                fn into_mut(self, scope: &mut #scope) -> Self {
                    Self {
                        #(#body),*
                    }
                }
            }

            impl #generics #debug for #name_expand #generic_names #where_clause {}
        }
    }
}

impl TypeField {
    pub fn expand_field(&self) -> TokenStream {
        let cube_type = prelude_type("CubeType");
        let vis = &self.vis;
        let name = self.ident.as_ref().unwrap();
        let ty = &self.ty;
        if self.comptime.is_present() {
            quote![#vis #name: #ty]
        } else {
            quote![#vis #name: <#ty as #cube_type>::ExpandType]
        }
    }

    pub fn clone_field(&self) -> TokenStream {
        let name = self.ident.as_ref().unwrap();
        quote![#name: self.#name.clone()]
    }

    pub fn launch_field(&self) -> TokenStream {
        let launch_arg = prelude_type("LaunchArg");
        let vis = &self.vis;
        let name = self.ident.as_ref().unwrap();
        let ty = &self.ty;

        if !self.comptime.is_present() {
            quote![#vis #name: <#ty as #launch_arg>::RuntimeArg<'a, R>]
        } else {
            quote![#vis #name: #ty]
        }
    }

    pub fn launch_new_arg(&self) -> TokenStream {
        let launch_arg = prelude_type("LaunchArg");
        let name = self.ident.as_ref().unwrap();
        let ty = &self.ty;

        if !self.comptime.is_present() {
            quote![#name: <#ty as #launch_arg>::RuntimeArg<'a, R>]
        } else {
            quote![#name: #ty]
        }
    }

    pub fn compilation_arg_field(&self) -> TokenStream {
        let launch_arg = prelude_type("LaunchArg");
        let vis = &self.vis;
        let name = self.ident.as_ref().unwrap();
        let ty = &self.ty;

        if !self.comptime.is_present() {
            quote![#vis #name: <#ty as #launch_arg>::CompilationArg]
        } else {
            quote![#vis #name: #ty]
        }
    }

    pub fn split(&self) -> (&Visibility, &Ident, &Type, bool) {
        (
            &self.vis,
            self.ident.as_ref().unwrap(),
            &self.ty,
            self.comptime.is_present(),
        )
    }
}
