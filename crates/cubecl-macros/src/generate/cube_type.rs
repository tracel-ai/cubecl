use darling::FromDeriveInput;
use proc_macro2::TokenStream;
use quote::quote;
use syn::{Ident, Type, Visibility};

use crate::{
    parse::cube_type::{TypeCodegen, TypeField},
    paths::{core_type, prelude_type},
};

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

    pub fn split(&self) -> (&Visibility, &Ident, &Type) {
        (&self.vis, self.ident.as_ref().unwrap(), &self.ty)
    }
}

impl TypeCodegen {
    pub fn expand_ty(&self) -> proc_macro2::TokenStream {
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

    pub fn launch_ty(&self) -> proc_macro2::TokenStream {
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

    pub fn launch_new(&self) -> proc_macro2::TokenStream {
        let args = self.fields.iter().map(TypeField::launch_field);
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

    pub fn arg_settings_impl(&self) -> proc_macro2::TokenStream {
        let arg_settings = prelude_type("ArgSettings");
        let kernel_launcher = prelude_type("KernelLauncher");
        let kernel_settings = core_type("KernelSettings");
        let name = &self.name_launch;
        let register_body = self
            .fields
            .iter()
            .map(TypeField::split)
            .map(|(_, ident, _)| quote![self.#ident.register(launcher)]);
        let config_input_body = self.fields.iter().enumerate().map(|(pos, field)| {
            let ident = &field.ident;
            quote![settings = #arg_settings::<R>::configure_input(&self.#ident, #pos, settings)]
        });
        let config_output_body = self.fields.iter().enumerate().map(|(pos, field)| {
            let ident = &field.ident;
            quote![settings = #arg_settings::<R>::configure_output(&self.#ident, #pos, settings)]
        });

        let generics = self.expanded_generics();
        let (generics, generic_names, where_clause) = generics.split_for_impl();

        quote! {
            impl #generics #arg_settings<R> for #name #generic_names #where_clause {
                fn register(&self, launcher: &mut #kernel_launcher<R>) {
                    #(#register_body;)*
                }

                fn configure_input(&self, position: usize, mut settings: #kernel_settings) -> #kernel_settings {
                    #(#config_input_body;)*

                    settings
                }

                fn configure_output(&self, position: usize, mut settings: #kernel_settings) -> #kernel_settings {
                    #(#config_output_body;)*

                    settings
                }
            }
        }
    }

    pub fn cube_type_impl(&self) -> proc_macro2::TokenStream {
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

    pub fn launch_arg_impl(&self) -> proc_macro2::TokenStream {
        let launch_arg_expand = prelude_type("LaunchArgExpand");
        let body_input = self.fields.iter().map(TypeField::split).map(|(vis, name, ty)| {
            quote![#vis #name: <#ty as #launch_arg_expand>::expand(builder, vectorization)]
        });
        let body_output = self.fields.iter().map(TypeField::split).map(|(vis, name, ty)| {
            quote![#vis #name: <#ty as #launch_arg_expand>::expand_output(builder, vectorization)]
        });

        let name = &self.ident;
        let name_launch = &self.name_launch;
        let name_expand = &self.name_expand;

        let (type_generics, type_generic_names, where_clause) = self.generics.split_for_impl();

        let assoc_generics = self.assoc_generics();
        let all = self.expanded_generics();
        let (_, all_generic_names, _) = all.split_for_impl();

        quote! {
            impl #type_generics LaunchArg for #name #type_generic_names #where_clause {
                type RuntimeArg #assoc_generics = #name_launch #all_generic_names;
            }

            impl #type_generics LaunchArgExpand for #name #type_generic_names #where_clause {
                fn expand(
                    builder: &mut KernelBuilder,
                    vectorization: cubecl::ir::Vectorization,
                ) -> <Self as CubeType>::ExpandType {
                    #name_expand {
                        #(#body_input),*
                    }
                }
                fn expand_output(
                    builder: &mut KernelBuilder,
                    vectorization: cubecl::ir::Vectorization,
                ) -> <Self as CubeType>::ExpandType {
                    #name_expand {
                        #(#body_output),*
                    }
                }
            }
        }
    }

    pub fn expand_type_impl(&self) -> proc_macro2::TokenStream {
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
            .map(|(_, name, _)| quote![#name: self.#name.__expand_runtime_method(context)]);

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

pub(crate) fn generate_cube_type(ast: &syn::DeriveInput, with_launch: bool) -> TokenStream {
    let codegen = match TypeCodegen::from_derive_input(ast) {
        Ok(codegen) => codegen,
        Err(e) => return e.write_errors(),
    };

    let expand_ty = codegen.expand_ty();
    let launch_ty = codegen.launch_ty();
    let launch_new = codegen.launch_new();

    let cube_type_impl = codegen.cube_type_impl();
    let arg_settings_impl = codegen.arg_settings_impl();
    let launch_arg_impl = codegen.launch_arg_impl();
    let expand_type_impl = codegen.expand_type_impl();

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
