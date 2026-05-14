use core::iter;

use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::Ident;

use crate::{
    parse::cube_type::{CubeTypeEnum, CubeTypeVariant, VariantKind},
    paths::{frontend_type, prelude_type},
};

impl CubeTypeEnum {
    pub fn generate_runtime(&self, with_launch: bool) -> TokenStream {
        if let Err(err) = self.validate() {
            return err.into_compile_error();
        }

        if with_launch {
            let args_ty = self.args_ty();
            let args_ty_runtime = self.args_ty_runtime();
            let launch_arg_impl = self.launch_arg_impl_runtime();

            let compilation_arg_ty_runtime = self.compilation_arg_ty_runtime();

            quote! {
                #args_ty
                #args_ty_runtime
                #launch_arg_impl

                #compilation_arg_ty_runtime
            }
        } else {
            let expand_value_ty = self.expand_value_ty();
            let cube_type_impl = self.cube_type_impl_runtime();
            let expand_type_impl = self.expand_type_impl_runtime();

            quote! {
                #expand_value_ty
                #cube_type_impl
                #expand_type_impl
            }
        }
    }

    fn validate(&self) -> Result<(), syn::Error> {
        let types = self
            .variants
            .iter()
            .filter_map(|v| match v.kind {
                VariantKind::Named => Some(Err(syn::Error::new(
                    self.ident.span(),
                    "Named enum fields are not supported for runtime enums",
                ))),
                VariantKind::Unnamed if v.fields.len() > 1 => Some(Err(syn::Error::new(
                    self.ident.span(),
                    "Only single value is supported for runtime enums",
                ))),
                VariantKind::Unnamed => Some(Ok(v.fields.iter().next().unwrap().ty.clone())),
                VariantKind::Empty => None,
            })
            .collect::<Result<Vec<_>, _>>()?;
        if types.len() > 1 {
            Err(syn::Error::new(
                self.ident.span(),
                "Only one value type is allowed for runtime enums",
            ))
        } else {
            Ok(())
        }
    }

    fn expand_value_ty(&self) -> proc_macro2::TokenStream {
        let cube_enum = prelude_type("CubeEnum");
        let expand_elem = frontend_type("NativeExpand");

        let expand_derives = match &self.derive {
            Some(derives) => quote![#[#derives]],
            None => quote![],
        };

        let name_expand = &self.name_expand;
        let (generics, generic_names, where_clause) = self.generics.split_for_impl();
        let vis = &self.vis;

        quote! {
            #expand_derives
            #vis struct #name_expand #generics #where_clause {
                discriminant: #expand_elem<i32>,
                value: <#name_expand #generic_names as #cube_enum>::RuntimeValue,
            }
        }
    }

    fn expand_type_impl_runtime(&self) -> proc_macro2::TokenStream {
        let scope = prelude_type("Scope");
        let clone = prelude_type("ExpandTypeClone");
        let into_expand = prelude_type("IntoExpand");
        let into_mut = prelude_type("IntoMut");
        let as_ref = prelude_type("AsRefExpand");
        let as_mut = prelude_type("AsMutExpand");
        let debug = prelude_type("CubeDebug");

        let name = &self.ident;
        let name_expand = &self.name_expand;
        let (generics, generic_names, where_clause) = self.generics.split_for_impl();

        let constructors = if self.with_constructors {
            let new_variant_functions = self.variants.iter().map(|v| {
                v.new_variant_function_runtime(
                    v.discriminant,
                    name_expand,
                    &generic_names,
                    self.value_ty(),
                )
            });

            Some(quote! {
                #[allow(non_snake_case)]
                #[allow(unused)]
                impl #generics #name #generic_names #where_clause {
                    #(
                        #new_variant_functions
                    )*
                }
            })
        } else {
            None
        };

        quote! {
            impl #generics #into_expand for #name_expand #generic_names #where_clause {
                type Expand = Self;

                fn into_expand(self, _: &#scope) -> Self {
                    self
                }
            }

            impl #generics #into_mut for #name_expand #generic_names #where_clause {
                fn into_mut(mut self, scope: &#scope) -> Self {
                    Self {
                        discriminant: #into_mut::into_mut(self.discriminant, scope),
                        value: #into_mut::into_mut(self.value, scope)
                    }
                }
            }

            impl #generics #debug for #name #generic_names #where_clause {}
            impl #generics #debug for #name_expand #generic_names #where_clause {}
            impl #generics #as_ref for #name_expand #generic_names #where_clause {
                fn __expand_ref_method(&self, _: &#scope) -> &Self {
                    self
                }
            }
            impl #generics #as_mut for #name_expand #generic_names #where_clause {
                fn __expand_ref_mut_method(&mut self, _: &#scope) -> &mut Self {
                    self
                }
            }

            impl #generics #clone for #name_expand #generic_names #where_clause {
                fn clone_unchecked(&self) -> Self {
                    Self {
                        discriminant: self.discriminant.clone(),
                        value: #clone::clone_unchecked(&self.value)
                    }
                }
            }

            #constructors
        }
    }

    fn value_ty(&self) -> TokenStream {
        self.variants
            .iter()
            .find_map(|v| match v.kind {
                VariantKind::Named => unimplemented!(),
                VariantKind::Unnamed => Some(v.fields.iter().next().unwrap().ty.clone()),
                VariantKind::Empty => None,
            })
            .map(|ty| quote![#ty])
            .unwrap_or_else(|| quote![()])
    }

    fn cube_type_impl_runtime(&self) -> proc_macro2::TokenStream {
        let cube_type = prelude_type("CubeType");
        let cube_enum = prelude_type("CubeEnum");
        let expand_elem = frontend_type("NativeExpand");

        let name = &self.ident;
        let name_expand = &self.name_expand;

        let (generics, generic_names, where_clause) = self.generics.split_for_impl();

        let value_ty = self.value_ty();

        let body_discriminants = self.match_impl(
            quote! {variant_name},
            self.variants
                .iter()
                .map(|v| {
                    let name = v.ident.to_string();
                    let discriminant = v.discriminant;
                    quote![#name => #discriminant]
                })
                .chain(iter::once(quote![_ => unreachable!()]))
                .collect(),
        );

        quote! {
            impl #generics #cube_type for #name #generic_names #where_clause {
                type ExpandType = #name_expand #generic_names;
            }
            impl #generics #cube_enum for #name_expand #generic_names #where_clause {
                type RuntimeValue = <#value_ty as #cube_type>::ExpandType;

                fn discriminant(&self) -> #expand_elem<i32> {
                    self.discriminant.clone()
                }

                fn runtime_value(self) -> Self::RuntimeValue {
                    self.value
                }

                fn discriminant_of(variant_name: &'static str) -> i32 {
                    #body_discriminants
                }
            }
        }
    }

    fn args_ty_runtime(&self) -> proc_macro2::TokenStream {
        let launch_name = format_ident!("{}Launch", self.ident);
        let args_name = format_ident!("{}Args", self.ident);
        let vis = &self.vis;

        let generics = self.expanded_generics();
        let (_, generic_names, _) = generics.split_for_impl();
        let where_clause = self.launch_arg_where();

        quote! {
            #vis enum #launch_name #generics #where_clause {
                Comptime(#args_name #generic_names),
                Runtime(#args_name #generic_names)
            }
        }
    }

    fn compilation_arg_ty_runtime(&self) -> proc_macro2::TokenStream {
        let launch_arg = prelude_type("LaunchArg");

        let name = Ident::new(&format!("{}CompilationArg", self.ident), Span::call_site());
        let vis = &self.vis;

        let generics = &self.generics;
        let value_ty = self.value_ty();

        let (generics_impl, generic_names, _) = self.generics.split_for_impl();
        let where_clause = self.launch_arg_where();

        let body_debug_discriminant = self.match_impl(
            quote! {discriminant},
            self.variants
                .iter()
                .map(|v| {
                    let discriminant = v.discriminant;
                    let name = v.ident.to_string();
                    quote![#discriminant => #name]
                })
                .chain(iter::once(quote![_ => unreachable!()]))
                .collect(),
        );

        quote! {
            #vis enum #name #generics #where_clause {
                Comptime {
                    discriminant: i32,
                    value: <#value_ty as #launch_arg>::CompilationArg
                },
                Runtime {
                    discriminant: <i32 as #launch_arg>::CompilationArg,
                    value: <#value_ty as #launch_arg>::CompilationArg
                }
            }


            impl #generics_impl Clone for #name #generic_names #where_clause {
                fn clone(&self) -> Self {
                    match self {
                        Self::Comptime { discriminant, value } => Self::Comptime {
                            discriminant: *discriminant,
                            value: value.clone()
                        },
                        Self::Runtime { discriminant, value } => Self::Runtime {
                            discriminant: discriminant.clone(),
                            value: value.clone()
                        }
                    }
                }
            }

            impl #generics_impl PartialEq for #name #generic_names #where_clause {
                fn eq(&self, other: &Self) -> bool {
                    match (self, other) {
                        (
                            Self::Comptime { discriminant: discriminant_this, value: value_this },
                            Self::Comptime { discriminant: discriminant_other, value: value_other }
                        ) => {
                            discriminant_this == discriminant_other && value_this == value_other
                        }
                        (
                            Self::Runtime { discriminant: discriminant_this, value: value_this },
                            Self::Runtime { discriminant: discriminant_other, value: value_other }
                        ) => {
                            discriminant_this == discriminant_other && value_this == value_other
                        }
                        _ => false
                    }
                }
            }

            impl #generics_impl Eq for #name #generic_names #where_clause {}

            impl #generics_impl core::hash::Hash for #name #generic_names #where_clause {
                fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
                    match self {
                        Self::Comptime { discriminant, value } => {
                            discriminant.hash(state);
                            value.hash(state);
                        }
                        Self::Runtime { discriminant, value } => {
                            discriminant.hash(state);
                            value.hash(state);
                        }
                    }
                }
            }

            impl #generics_impl core::fmt::Debug for #name #generic_names #where_clause {
                fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                    match self {
                        Self::Comptime { discriminant, value } => {
                            f.debug_struct(stringify!(#name))
                                .field(stringify!(discriminant), &cubecl::format::DebugRaw(#body_debug_discriminant))
                                .field(stringify!(value), value)
                                .finish()
                        }
                        Self::Runtime { discriminant, value } => {
                            f.debug_struct(stringify!(#name))
                                .field(stringify!(discriminant), discriminant)
                                .field(stringify!(value), value)
                                .finish()
                        }
                    }
                }
            }
        }
    }

    fn register_impl_runtime(&self) -> proc_macro2::TokenStream {
        let launch_arg = prelude_type("LaunchArg");
        let kernel_launcher = prelude_type("KernelLauncher");
        let args_name = Ident::new(&format!("{}Args", self.ident), Span::call_site());
        let compilation_arg_name =
            Ident::new(&format!("{}CompilationArg", self.ident), Span::call_site());
        let launch_name = Ident::new(&format!("{}Launch", self.ident), Span::call_site());

        let (_, generic_names, _) = self.generics.split_for_impl();

        let value_ty = self.value_ty();

        let body_discriminant = self.match_impl(
            quote! {value},
            self.variants
                .iter()
                .map(|v| {
                    let discriminant = v.discriminant;
                    let name = &v.ident;
                    let pat = match v.kind {
                        VariantKind::Named => quote![#args_name::#name { .. }],
                        VariantKind::Unnamed => quote![#args_name::#name (..)],
                        VariantKind::Empty => quote![#args_name::#name],
                    };

                    quote![#pat => #discriminant]
                })
                .collect(),
        );

        let register_value = self.match_impl(
            quote! {value},
            self.variants
                .iter()
                .map(|v| {
                    let name = &v.ident;
                    match v.kind {
                        VariantKind::Named => unimplemented!(),
                        VariantKind::Unnamed => {
                            quote![#args_name::#name (value) => <#value_ty as #launch_arg>::register(value, launcher)]
                        }
                        VariantKind::Empty => {
                            quote![#args_name::#name => <#value_ty as #launch_arg>::register(Default::default(), launcher)]
                        }
                    }
                })
                .collect(),
        );

        quote! {
            fn register<R: Runtime>(runtime_arg: Self::RuntimeArg<R>, launcher: &mut #kernel_launcher<R>) -> Self::CompilationArg {
                match runtime_arg {
                    #launch_name::Comptime(value) => {
                        let discriminant = #body_discriminant;
                        <i32 as #launch_arg>::register(discriminant, launcher);
                        let value = #register_value;
                        #compilation_arg_name #generic_names :: Comptime {
                            discriminant,
                            value
                        }
                    }
                    #launch_name::Runtime(value) => {
                        let discriminant = #body_discriminant;
                        let discriminant = <i32 as #launch_arg>::register(discriminant, launcher);
                        let value = #register_value;
                        #compilation_arg_name #generic_names :: Runtime {
                            discriminant,
                            value
                        }
                    }
                }
            }
        }
    }

    fn launch_arg_impl_runtime(&self) -> proc_macro2::TokenStream {
        let launch_arg = prelude_type("LaunchArg");
        let cube_type = prelude_type("CubeType");
        let kernel_builder = prelude_type("KernelBuilder");
        let name = &self.ident;
        let name_launch = Ident::new(&format!("{}Launch", self.ident), Span::call_site());
        let compilation_arg =
            Ident::new(&format!("{}CompilationArg", self.ident), Span::call_site());
        let expand_name = &self.name_expand;

        let (generics, generic_names, _) = self.generics.split_for_impl();
        let where_clause = self.launch_arg_where();

        let assoc_generics = self.assoc_generics();
        let all = self.expanded_generics();
        let (_, all_generic_names, _) = all.split_for_impl();

        let register_impl_runtime = self.register_impl_runtime();

        let value_ty = self.value_ty();

        quote! {
            impl #generics #launch_arg for #name #generic_names #where_clause {
                type RuntimeArg #assoc_generics = #name_launch #all_generic_names;
                type CompilationArg = #compilation_arg #generic_names;

                #register_impl_runtime

                fn expand(arg: &Self::CompilationArg, builder: &mut #kernel_builder) -> <Self as #cube_type>::ExpandType {
                    match arg {
                        #compilation_arg::Comptime { discriminant, value } => {
                            let value = <#value_ty as #launch_arg>::expand(value, builder);
                            #expand_name #generic_names {
                                discriminant: (*discriminant).into(),
                                value,
                            }
                        }
                        #compilation_arg::Runtime { discriminant, value } => {
                            let discriminant = <i32 as #launch_arg>::expand(discriminant, builder);
                            let value = <#value_ty as #launch_arg>::expand(value, builder);
                            #expand_name #generic_names {
                                discriminant,
                                value,
                            }
                        }
                    }
                }
            }
        }
    }
}

impl CubeTypeVariant {
    fn new_variant_function_runtime(
        &self,
        index: i32,
        ident_ty_expand: &Ident,
        generics: &syn::TypeGenerics,
        value_ty: TokenStream,
    ) -> TokenStream {
        let scope = prelude_type("Scope");
        let cube_type = prelude_type("CubeType");
        let into_runtime = prelude_type("IntoRuntime");
        let ident = &self.ident;
        let base_function = Ident::new(&format!("new_{ident}"), ident.span());
        let expand_function = Ident::new(&format!("__expand_new_{ident}"), ident.span());

        match self.kind {
            VariantKind::Named => {
                unimplemented!("Not supported for now")
            }
            VariantKind::Unnamed => {
                let ty = self.fields.iter().next().unwrap().ty.clone();

                quote! {
                    pub fn #base_function(value: #ty) -> Self {
                        cubecl::unexpanded!()
                    }

                    pub fn #expand_function(_: &#scope, value: <#ty as #cube_type>::ExpandType) -> #ident_ty_expand #generics {
                        #ident_ty_expand #generics {
                            discriminant: #index.into(),
                            value
                        }
                    }
                }
            }
            VariantKind::Empty => {
                quote! {
                    pub fn #base_function() -> Self {
                        cubecl::unexpanded!()
                    }

                    pub fn #expand_function(scope: &#scope) -> #ident_ty_expand #generics {
                        #ident_ty_expand #generics {
                            discriminant: #index.into(),
                            value: <#value_ty as #into_runtime>::__expand_runtime_method(Default::default(), scope),
                        }
                    }
                }
            }
        }
    }
}
