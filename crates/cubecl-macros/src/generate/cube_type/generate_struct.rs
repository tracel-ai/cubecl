use proc_macro2::TokenStream;
use quote::quote;
use syn::{Ident, Type, Visibility};

use crate::{
    parse::cube_type::{CubeTypeStruct, TypeField},
    paths::prelude_type,
};

impl CubeTypeStruct {
    pub fn generate(&self, with_launch: bool) -> TokenStream {
        let expand_ty = self.expand_ty();
        let launch_ty = self.launch_ty();
        let launch_new = self.launch_new();

        let cube_type_impl = self.cube_type_impl();
        let arg_settings_impl = self.arg_settings_impl();
        let launch_arg_impl = self.launch_arg_impl();
        let expand_type_impl = self.expand_type_impl();

        if with_launch {
            quote! {
                #expand_ty
                #launch_ty
                #launch_new

                #cube_type_impl
                #arg_settings_impl
                #launch_arg_impl
                #expand_type_impl
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
        let fields = self.fields.iter().map(TypeField::expand_field);
        let name = &self.name_expand;
        let generics = &self.generics;
        let vis = &self.vis;

        quote! {
            #[derive(Clone)]
            #vis struct #name #generics {
                #(#fields),*
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
            .map(TypeField::split)
            .map(|(_, ident, _)| quote![self.#ident.register(launcher)]);

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

            quote! {
               #name: <#ty as #launch_arg>::compilation_arg::<R>(&runtime_arg.#name)
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

        fn gen<F: Fn(&Ident) -> TokenStream>(fields: &[TypeField], func: F) -> Vec<TokenStream> {
            fields
                .iter()
                .map(|field| func(field.ident.as_ref().unwrap()))
                .collect::<Vec<_>>()
        }
        let clone = gen(&self.fields, |name| quote!(#name: self.#name.clone()));
        let hash = gen(&self.fields, |name| quote!(self.#name.hash(state)));
        let partial_eq = gen(&self.fields, |name| quote!(self.#name.eq(&other.#name)));
        let debug = gen(&self.fields, |name| {
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
        let launch_arg_expand = prelude_type("LaunchArgExpand");
        let body_input = self.fields.iter().map(TypeField::split).map(|(_vis, name, ty)| {
            quote![#name: <#ty as #launch_arg_expand>::expand(&arg.#name, builder)]
        });
        let body_output = self.fields.iter().map(TypeField::split).map(|(_vis, name, ty)| {
            quote![#name: <#ty as #launch_arg_expand>::expand_output(&arg.#name, builder)]
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

            impl #type_generics LaunchArg for #name #type_generic_names #where_clause {
                type RuntimeArg #assoc_generics = #name_launch #all_generic_names;

                fn compilation_arg<'a, R: Runtime>(
                    runtime_arg: &Self::RuntimeArg<'a, R>,
                ) -> Self::CompilationArg {
                    #compilation_arg_impl
                }
            }

            impl #type_generics LaunchArgExpand for #name #type_generic_names #where_clause {
                type CompilationArg = #compilation_ident #compilation_generics;

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
        let init = prelude_type("Init");
        let into_runtime = prelude_type("IntoRuntime");
        let context = prelude_type("CubeContext");
        let name = &self.ident;
        let name_expand = &self.name_expand;
        let (generics, generic_names, where_clause) = self.generics.split_for_impl();
        let body = self
            .fields
            .iter()
            .map(TypeField::split)
            .map(|(_, ident, _)| quote![#ident: #init::init(self.#ident, context)]);
        let fields_to_runtime = self
            .fields
            .iter()
            .map(TypeField::split)
            .map(|(_, name, _)| quote![#name: #into_runtime::__expand_runtime_method(self.#name, context)]);

        quote! {
            impl #generics #init for #name_expand #generic_names #where_clause {
                fn init(self, context: &mut #context) -> Self {
                    Self {
                        #(#body),*
                    }
                }
            }

            impl #generics #into_runtime for #name #generic_names #where_clause {
                fn __expand_runtime_method(self, context: &mut CubeContext) -> Self::ExpandType {
                    let expand = #name_expand {
                        #(#fields_to_runtime),*
                    };
                    Init::init(expand, context)
                }
            }
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

    pub fn launch_field(&self) -> TokenStream {
        let launch_arg = prelude_type("LaunchArg");
        let vis = &self.vis;
        let name = self.ident.as_ref().unwrap();
        let ty = &self.ty;
        quote![#vis #name: <#ty as #launch_arg>::RuntimeArg<'a, R>]
    }

    pub fn launch_new_arg(&self) -> TokenStream {
        let launch_arg = prelude_type("LaunchArg");
        let name = self.ident.as_ref().unwrap();
        let ty = &self.ty;
        quote![#name: <#ty as #launch_arg>::RuntimeArg<'a, R>]
    }

    pub fn compilation_arg_field(&self) -> TokenStream {
        let launch_arg = prelude_type("LaunchArgExpand");
        let vis = &self.vis;
        let name = self.ident.as_ref().unwrap();
        let ty = &self.ty;
        quote![#vis #name: <#ty as #launch_arg>::CompilationArg]
    }

    pub fn split(&self) -> (&Visibility, &Ident, &Type) {
        (&self.vis, self.ident.as_ref().unwrap(), &self.ty)
    }
}
