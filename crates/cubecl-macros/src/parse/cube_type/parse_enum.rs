use darling::FromDeriveInput;
use syn::{spanned::Spanned, Ident};

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
                    .map(|a| CubeTypeVariant {
                        ident: a.ident.clone(),
                        fields: a.fields.clone(),
                    })
                    .collect(),
            }),
            _ => Err(darling::Error::custom("Only enum are supported.")),
        }
    }
}
