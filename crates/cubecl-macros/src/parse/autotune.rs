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
    Default,
    Exp(AnchorExp),
}

#[derive(FromMeta)]
pub struct AnchorExp {
    max: Option<usize>,
    min: Option<usize>,
    base: Option<usize>,
}

impl Anchor {
    pub fn max(&self) -> TokenStream {
        match self {
            Self::Exp(val) => val.max(),
            Self::Default => quote![None],
        }
    }
    pub fn min(&self) -> TokenStream {
        match self {
            Self::Exp(val) => val.min(),
            Self::Default => quote![None],
        }
    }
    pub fn base(&self) -> TokenStream {
        match self {
            Self::Exp(val) => val.base(),
            Self::Default => quote![None],
        }
    }
}
impl AnchorExp {
    pub fn max(&self) -> TokenStream {
        match self.max {
            Some(val) => quote![Some(#val)],
            None => quote![None],
        }
    }
    pub fn min(&self) -> TokenStream {
        match self.min {
            Some(val) => quote![Some(#val)],
            None => quote![None],
        }
    }
    pub fn base(&self) -> TokenStream {
        match self.base {
            Some(val) => quote![Some(#val)],
            None => quote![None],
        }
    }
}
