use darling::FromDeriveInput;
use ident_case::RenameRule;
use proc_macro2::TokenStream;
use quote::{ToTokens, format_ident, quote};
use syn::{DeriveInput, Index, Member};

use crate::{
    parse::autotune::{AutotuneKey, AutotuneKeyField},
    paths::tune_type,
};

impl AutotuneKey {
    fn generate_fmt_str(&self) -> String {
        let name = self.ident.to_string();
        let fields = self.data.as_ref().take_struct().unwrap();
        if self.is_tuple() {
            let fields: Vec<&str> = fields.iter().map(|_| "{:?}").collect();
            let fields = fields.join(", ");
            format!("{name}({fields})")
        } else {
            let fields: Vec<String> = fields
                .iter()
                .map(|field| {
                    let name = field.name.clone().unwrap_or_else(|| {
                        RenameRule::PascalCase
                            .apply_to_field(field.ident.as_ref().unwrap().to_string())
                    });

                    format!("{name}: {{:?}}")
                })
                .collect();
            let fields = fields.join(", ");
            format!("{name} - {fields}")
        }
    }

    fn generate_fmt(&self) -> TokenStream {
        let name = &self.ident;
        let (generics, generic_names, where_clause) = self.generics.split_for_impl();
        let fmt_str = self.generate_fmt_str();
        let fields = self.data.as_ref().take_struct().unwrap();
        let fmt_args = fields.iter().enumerate().map(|(i, field)| {
            if let Some(ident) = field.ident.as_ref() {
                quote![self.#ident]
            } else {
                let idx = Index::from(i);
                quote![self.#idx]
            }
        });
        let fmt_call = quote![write!(f, #fmt_str, #(#fmt_args),*)];
        quote! {
            impl #generics ::core::fmt::Display for #name #generic_names #where_clause {
                fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                    #fmt_call
                }
            }
        }
    }

    fn generate_new(&self) -> TokenStream {
        let name = &self.ident;
        let (generics, generic_names, where_clause) = self.generics.split_for_impl();
        let vis = &self.vis;
        let new_fn = match self.is_tuple() {
            true => self.generate_new_fn_tuple(),
            false => self.generate_new_fn_named(),
        };

        quote! {
            impl #generics #name #generic_names #where_clause {
                #[allow(clippy::too_many_arguments, missing_docs)]
                #vis #new_fn
            }
        }
    }

    fn generate_new_fn_named(&self) -> TokenStream {
        let fields = self.data.as_ref().take_struct().unwrap();
        let new_args = fields.iter().map(|it| {
            let name = it.ident.as_ref().unwrap();
            let ty = &it.ty;
            quote![#name: #ty]
        });
        let field_inits = fields.iter().map(|field| {
            let name = field.ident.as_ref().unwrap();
            let init = field_init(field, Member::Named(name.clone()));
            quote![#name: #init]
        });

        quote! {
            fn new(#(#new_args),*) -> Self {
                Self {
                    #(#field_inits),*
                }
            }
        }
    }

    fn generate_new_fn_tuple(&self) -> TokenStream {
        let fields = self.data.as_ref().take_struct().unwrap();
        let new_args = fields.iter().enumerate().map(|(i, field)| {
            let name = format_ident!("{i}_");
            let ty = &field.ty;
            quote![#name: #ty]
        });
        let field_inits = fields
            .iter()
            .enumerate()
            .map(|(i, field)| field_init(field, Member::Unnamed(Index::from(i))));

        quote! {
            fn new(#(#new_args),*) -> Self {
                Self (
                    #(#field_inits),*
                )
            }
        }
    }

    fn generate_key_impl(&self) -> TokenStream {
        let key = tune_type("AutotuneKey");
        let name = &self.ident;
        let (generics, generic_names, where_clause) = self.generics.split_for_impl();
        quote![impl #generics #key for #name #generic_names #where_clause {}]
    }
}

fn field_init(field: &AutotuneKeyField, member: Member) -> TokenStream {
    let anchor_fn = tune_type("anchor");
    field
        .anchor
        .as_ref()
        .map(|anchor| {
            let max = anchor.max();
            let min = anchor.min();
            let base = anchor.base();

            quote![#anchor_fn(#member, #max, #min, #base)]
        })
        .unwrap_or_else(|| member.to_token_stream())
}

pub fn generate_autotune_key(input: DeriveInput) -> syn::Result<TokenStream> {
    let key = AutotuneKey::from_derive_input(&input)?;
    let display = key.generate_fmt();
    let new = key.generate_new();
    let key_impl = key.generate_key_impl();
    Ok(quote! {
        #display
        #new
        #key_impl
    })
}
