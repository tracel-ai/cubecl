use std::iter;

use darling::FromDeriveInput;
use proc_macro2::Span;
use syn::{Generics, Ident, parse_quote, punctuated::Punctuated, spanned::Spanned};

use crate::paths::prelude_type;

#[derive(Debug)]
pub struct CubeTypeEnum {
    pub ident: Ident,
    pub name_expand: Ident,
    pub variants: Vec<CubeTypeVariant>,
    pub generics: syn::Generics,
    pub vis: syn::Visibility,
}

#[derive(Debug)]
pub struct CubeTypeVariant {
    pub ident: Ident,
    pub fields: syn::Fields,
    pub field_names: Vec<Ident>,
    pub kind: VariantKind,
}

#[derive(Debug)]
pub enum VariantKind {
    Named,
    Unnamed,
    Empty,
}

impl FromDeriveInput for CubeTypeEnum {
    fn from_derive_input(input: &syn::DeriveInput) -> darling::Result<Self> {
        match &input.data {
            syn::Data::Enum(data) => Ok(Self {
                ident: input.ident.clone(),
                generics: input.generics.clone(),
                vis: input.vis.clone(),
                name_expand: Ident::new(format!("{}Expand", input.ident).as_str(), input.span()),
                variants: data
                    .variants
                    .iter()
                    .map(|a| {
                        let mut kind = if a.fields.is_empty() {
                            VariantKind::Empty
                        } else {
                            VariantKind::Unnamed
                        };

                        for field in a.fields.iter() {
                            if field.ident.is_some() {
                                kind = VariantKind::Named;
                            }
                        }

                        CubeTypeVariant {
                            kind,
                            ident: a.ident.clone(),
                            field_names: a
                                .fields
                                .iter()
                                .enumerate()
                                .map(|(i, field)| match &field.ident {
                                    Some(name) => name.clone(),
                                    None => {
                                        Ident::new(format!("arg_{i}").as_str(), Span::call_site())
                                    }
                                })
                                .collect(),
                            fields: a.fields.clone(),
                        }
                    })
                    .collect(),
            }),
            _ => Err(darling::Error::custom("Only enum are supported.")),
        }
    }
}

impl CubeTypeEnum {
    pub fn expanded_generics(&self) -> Generics {
        let runtime = prelude_type("Runtime");
        let mut generics = self.generics.clone();
        generics.params.push(parse_quote![R: #runtime]);
        let all = iter::once(parse_quote!['a]).chain(generics.params);
        generics.params = Punctuated::from_iter(all);
        generics
    }

    pub fn assoc_generics(&self) -> Generics {
        let runtime = prelude_type("Runtime");
        parse_quote![<'a, R: #runtime>]
    }
}
