use std::iter;

use darling::{FromDeriveInput, util::Flag};
use proc_macro2::Span;
use quote::format_ident;
use syn::{Generics, Ident, parse_quote, punctuated::Punctuated};

use crate::paths::prelude_type;

#[derive(Debug)]
pub struct CubeTypeEnum {
    pub ident: Ident,
    pub name_expand: Ident,
    pub variants: Vec<CubeTypeVariant>,
    pub generics: syn::Generics,
    pub vis: syn::Visibility,
    pub runtime_variants: bool,
    pub with_constructors: bool,
}

#[derive(Debug, FromDeriveInput)]
#[darling(attributes(cube), supports(enum_any))]
pub struct CubeTypeEnumRepr {
    ident: Ident,
    vis: syn::Visibility,
    generics: syn::Generics,
    data: darling::ast::Data<syn::Variant, ()>,
    runtime_variants: Flag,
    /// Don't generate constructors, useful for expanding existing types where a new impl isn't allowed
    no_constructors: Flag,
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
        let repr = CubeTypeEnumRepr::from_derive_input(input)?;
        match &repr.data {
            darling::ast::Data::Enum(variants) => Ok(Self {
                name_expand: format_ident!("{}Expand", repr.ident),
                ident: repr.ident,
                generics: repr.generics,
                vis: repr.vis,
                runtime_variants: repr.runtime_variants.is_present(),
                with_constructors: !repr.no_constructors.is_present(),
                variants: variants
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
        if !self.is_empty() {
            generics.params.push(parse_quote![R: #runtime]);
            let all = iter::once(parse_quote!['a]).chain(generics.params);
            generics.params = Punctuated::from_iter(all);
        }
        generics
    }

    pub fn arg_settings_generics(&self) -> Generics {
        let runtime = prelude_type("Runtime");
        let mut generics = self.generics.clone();
        generics.params.push(parse_quote![R: #runtime]);

        if !self.is_empty() {
            let all = iter::once(parse_quote!['a]).chain(generics.params);
            generics.params = Punctuated::from_iter(all);
        }
        generics
    }

    pub fn assoc_generics(&self) -> Generics {
        let runtime = prelude_type("Runtime");
        parse_quote![<'a, R: #runtime>]
    }

    pub fn is_empty(&self) -> bool {
        self.variants
            .iter()
            .all(|it| matches!(it.kind, VariantKind::Empty))
    }
}
