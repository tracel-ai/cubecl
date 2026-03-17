use darling::{
    FromDeriveInput, FromField, FromVariant,
    ast::{Data, Fields},
    uses_type_params,
    util::Flag,
};
use syn::{Generics, Ident, Type};

use crate::generate::RuntimeField;

#[derive(FromDeriveInput)]
#[darling(attributes(cube), allow_unknown_fields)]
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

#[derive(FromField, Clone)]
#[darling(attributes(cube))]
pub struct IntoRuntimeField {
    pub ident: Option<Ident>,
    pub ty: Type,
    pub comptime: Flag,
}

uses_type_params!(IntoRuntimeField, ty);
impl RuntimeField for IntoRuntimeField {
    fn ty(self) -> Type {
        self.ty
    }
}
