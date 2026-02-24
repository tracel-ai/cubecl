use darling::{
    FromDeriveInput, FromField, FromVariant,
    ast::{Data, Fields},
    util::Flag,
};
use syn::{Generics, Ident, Type};

#[derive(FromDeriveInput)]
#[darling(attributes(cube))]
pub struct IntoRuntime {
    pub ident: Ident,
    pub generics: Generics,
    pub data: Data<IntoRuntimeVariant, IntoRuntimeField>,
    pub runtime_variants: Flag,
}

#[derive(FromVariant)]
pub struct IntoRuntimeVariant {
    pub ident: Ident,
    pub fields: Fields<IntoRuntimeField>,
}

#[derive(FromField)]
#[darling(attributes(cube))]
pub struct IntoRuntimeField {
    pub ident: Option<Ident>,
    pub ty: Type,
    pub comptime: Flag,
}
