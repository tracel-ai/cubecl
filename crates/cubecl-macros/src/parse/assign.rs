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
pub struct Assign {
    pub ident: Ident,
    pub generics: Generics,
    pub data: Data<AssignVariant, AssignField>,
    pub runtime_variants: Flag,
}

#[derive(FromVariant)]
pub struct AssignVariant {
    pub ident: Ident,
    pub fields: Fields<AssignField>,
}

#[derive(FromField, Clone)]
#[darling(attributes(cube))]
pub struct AssignField {
    pub ident: Option<Ident>,
    pub comptime: Flag,
    pub ty: Type,
}

uses_type_params!(AssignField, ty);
impl RuntimeField for AssignField {
    fn ty(self) -> Type {
        self.ty
    }
}
