use darling::{FromDeriveInput, FromField, FromMeta, ast::Data};
use proc_macro2::TokenStream;
use quote::quote;
use syn::{Generics, Ident, Type, Visibility};

#[derive(FromDeriveInput)]
#[darling(supports(struct_any))]
pub struct AutotuneKey {
    pub ident: Ident,
    pub vis: Visibility,
    pub generics: Generics,
    pub data: Data<(), AutotuneKeyField>,
}

impl AutotuneKey {
    pub fn is_tuple(&self) -> bool {
        self.data.as_ref().take_struct().unwrap().is_tuple()
    }
}

#[derive(FromField)]
#[darling(attributes(autotune))]
pub struct AutotuneKeyField {
    pub ident: Option<Ident>,
    pub ty: Type,
    pub anchor: Option<Anchor>,
    pub name: Option<String>,
}

#[derive(FromMeta)]
pub enum Anchor {
    #[darling(word)]
    Unlimited,
    Max(usize),
}

impl Anchor {
    pub fn max(&self) -> TokenStream {
        match self {
            Anchor::Unlimited => quote![None],
            Anchor::Max(value) => quote![Some(#value)],
        }
    }
}
