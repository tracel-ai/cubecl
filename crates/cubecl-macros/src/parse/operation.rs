use darling::{
    ast::{Data, Fields},
    util::Flag,
    FromDeriveInput, FromField, FromVariant,
};
use syn::{Generics, Ident, Type, Visibility};

#[derive(FromDeriveInput)]
#[darling(supports(enum_newtype), attributes(operation))]
pub struct Operation {
    pub ident: Ident,
    pub vis: Visibility,
    pub generics: Generics,
    pub data: Data<OperationVariant, ()>,

    pub opcode_name: Ident,
    pub with_children: Flag,
}

#[derive(FromVariant)]
pub struct OperationVariant {
    pub ident: Ident,
    pub fields: Fields<OperationField>,
}

#[derive(FromField)]
pub struct OperationField {
    pub ty: Type,
}
