use darling::{
    FromDeriveInput,
    ast::{Data, Style},
};
use proc_macro2::TokenStream;
use quote::{ToTokens, format_ident, quote};
use syn::{DeriveInput, Index, WhereClause};

use crate::{
    generate::bounded_where_clause,
    parse::assign::{Assign, AssignVariant},
    paths::prelude_type,
};

impl ToTokens for Assign {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let assign = prelude_type("Assign");
        let scope = prelude_type("Scope");

        let name = &self.ident;
        let expand_name = format_ident!("{name}Expand");
        let (generics, generic_names, _) = self.generics.split_for_impl();
        let where_clause = self.where_clause();

        let init_mut_body = match &self.data {
            Data::Enum(_) if self.runtime_variants.is_present() => {
                self.init_mut_body_runtime_enum()
            }
            Data::Enum(_) => self.init_mut_body_comptime_enum(),
            Data::Struct(_) => self.init_mut_body_struct(),
        };

        let assign_body = match &self.data {
            Data::Enum(_) if self.runtime_variants.is_present() => self.assign_body_runtime_enum(),
            Data::Enum(_) => self.assign_body_comptime_enum(),
            Data::Struct(_) => self.assign_body_struct(),
        };

        tokens.extend(quote! {
            impl #generics #assign for #expand_name #generic_names #where_clause {
                fn expand_assign(&mut self, scope: &mut Scope, value: Self) {
                    use #assign as _;
                    #assign_body
                }

                fn init_mut(&self, scope: &mut #scope) -> Self {
                    use #assign as _;
                    #init_mut_body
                }
            }
        });
    }
}

impl Assign {
    fn assign_body_struct(&self) -> TokenStream {
        let struct_ = self.data.as_ref().take_struct().unwrap();

        let fields = struct_.fields.iter().enumerate().map(|(i, field)| {
            let index = Index::from(i);
            match &field.ident {
                Some(name) if field.comptime.is_present() => quote![self.#name = value.#name;],
                Some(name) => {
                    quote![self.#name.expand_assign(scope, value.#name);]
                }
                None if field.comptime.is_present() => quote![self.#index = value.#index;],
                None => quote![self.#index.expand_assign(scope, value.#index);],
            }
        });
        quote![#(#fields)*]
    }

    fn init_mut_body_struct(&self) -> TokenStream {
        let struct_ = self.data.as_ref().take_struct().unwrap();

        let fields = struct_.fields.iter().enumerate().map(|(i, field)| {
            let index = Index::from(i);
            match &field.ident {
                Some(name) => {
                    quote![#name: self.#name.init_mut(scope)]
                }
                None => quote![self.#index.init_mut(scope)],
            }
        });

        match struct_.style {
            Style::Tuple => quote![Self(#(#fields,)*)],
            Style::Struct => quote![Self { #(#fields,)* }],
            Style::Unit => quote![Self],
        }
    }

    fn assign_body_comptime_enum(&self) -> TokenStream {
        let variants = self.data.as_ref().take_enum().unwrap();

        let branches = variants
            .iter()
            .map(|variant| self.assign_body_comptime_variant(variant));

        quote![match self {
            #(#branches,)*
            _ => core::fmt::panic!("Can't assign to mismatched enum variants"),
        }]
    }

    fn assign_body_comptime_variant(&self, variant: &AssignVariant) -> TokenStream {
        let variant_name = &variant.ident;

        match variant.fields.style {
            Style::Tuple => {
                let field_names_this =
                    (0..variant.fields.len()).map(|i| format_ident!("_{i}_this"));
                let field_names_other =
                    (0..variant.fields.len()).map(|i| format_ident!("_{i}_other"));

                let fields = variant.fields.iter().enumerate().map(|(i, field)| {
                    let name_this = format_ident!("_{i}_this");
                    let name_other = format_ident!("_{i}_other");
                    match field.comptime.is_present() {
                        true => quote![*#name_this = #name_other;],
                        false => quote![#name_this.expand_assign(scope, #name_other);],
                    }
                });

                quote! {
                    (Self::#variant_name(#(#field_names_this,)*),
                        Self::#variant_name(#(#field_names_other,)*)
                    ) => {
                        #(#fields)*
                    }
                }
            }
            Style::Struct => {
                let field_names_this = variant.fields.iter().map(|field| {
                    let name = field.ident.as_ref().unwrap();
                    let rename = format_ident!("{name}_this");
                    quote![#name: #rename]
                });
                let field_names_other = variant.fields.iter().map(|field| {
                    let name = field.ident.as_ref().unwrap();
                    let rename = format_ident!("{name}_other");
                    quote![#name: #rename]
                });

                let fields = variant.fields.iter().map(|field| {
                    let name = field.ident.as_ref().unwrap();
                    let name_this = format_ident!("{name}_this");
                    let name_other = format_ident!("{name}_other");
                    match field.comptime.is_present() {
                        true => quote![*#name_this = #name_other;],
                        false => quote![#name_this.expand_assign(scope, #name_other);],
                    }
                });

                quote! {
                    (Self::#variant_name { #(#field_names_this,)* },
                        Self::#variant_name { #(#field_names_other,)* }
                    ) => {
                        #(#fields)*
                    }
                }
            }
            Style::Unit => {
                quote![(Self::#variant_name, Self::variant_name) => {}]
            }
        }
    }

    fn init_mut_body_comptime_enum(&self) -> TokenStream {
        let variants = self.data.as_ref().take_enum().unwrap();

        let branches = variants
            .iter()
            .map(|variant| self.init_mut_body_comptime_variant(variant));

        quote![match self {
            #(#branches,)*
        }]
    }

    fn init_mut_body_comptime_variant(&self, variant: &AssignVariant) -> TokenStream {
        let variant_name = &variant.ident;

        let field_names = variant
            .fields
            .iter()
            .enumerate()
            .map(|(i, field)| field.ident.clone().unwrap_or_else(|| format_ident!("_{i}")));

        let fields = variant.fields.iter().enumerate().map(|(i, field)| {
            let index = format_ident!("_{i}");
            match &field.ident {
                Some(name) => {
                    quote![#name: #name.init_mut(scope)]
                }
                None => quote![#index.init_mut(scope)],
            }
        });

        match variant.fields.style {
            Style::Tuple => {
                quote![Self::#variant_name(#(#field_names,)*) => Self::#variant_name(#(#fields,)*)]
            }
            Style::Struct => {
                quote![Self::#variant_name { #(#field_names,)* } => Self::#variant_name{ #(#fields,)* }]
            }
            Style::Unit => quote![Self::#variant_name => Self::#variant_name],
        }
    }

    fn assign_body_runtime_enum(&self) -> TokenStream {
        quote! {
            self.discriminant.expand_assign(scope, value.discriminant);
            self.value.expand_assign(scope, value.value);
        }
    }

    fn init_mut_body_runtime_enum(&self) -> TokenStream {
        quote! {
            Self {
                discriminant: self.discriminant.init_mut(scope),
                value: self.value.init_mut(scope),
            }
        }
    }

    fn where_clause(&self) -> Option<WhereClause> {
        let cube_type = prelude_type("CubeType");
        let assign = prelude_type("Assign");

        let fields: Vec<_> = match &self.data {
            Data::Enum(variants) => variants
                .iter()
                .flat_map(|it| it.fields.fields.iter())
                .filter(|it| !it.comptime.is_present())
                .cloned()
                .collect(),
            Data::Struct(fields) => fields
                .iter()
                .filter(|it| !it.comptime.is_present())
                .cloned()
                .collect(),
        };
        bounded_where_clause(
            &self.generics,
            fields,
            |param| quote!(<#param as #cube_type>::ExpandType: #assign),
        )
    }
}

pub fn generate_cube_type_mut(input: &DeriveInput) -> syn::Result<TokenStream> {
    let assign = Assign::from_derive_input(input)?;
    Ok(assign.into_token_stream())
}
