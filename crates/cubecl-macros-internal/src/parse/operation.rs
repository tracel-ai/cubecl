use darling::{
    FromDeriveInput, FromField, FromVariant,
    ast::{Data, Fields},
    util::Flag,
};
use syn::{Generics, Ident, Type, Visibility};

#[derive(FromDeriveInput)]
#[darling(supports(enum_newtype, enum_unit, enum_named), attributes(operation))]
pub struct Operation {
    pub ident: Ident,
    pub vis: Visibility,
    pub generics: Generics,
    pub data: Data<OperationVariant, ()>,

    pub opcode_name: Ident,
    pub commutative: Flag,
    pub pure: Flag,
}

#[derive(FromDeriveInput)]
#[darling(supports(enum_any), attributes(operation))]
pub struct OpCode {
    pub ident: Ident,
    pub vis: Visibility,
    pub generics: Generics,
    pub data: Data<OperationVariant, ()>,

    pub opcode_name: Ident,
}

#[derive(FromVariant)]
#[darling(attributes(operation))]
pub struct OperationVariant {
    pub ident: Ident,
    pub fields: Fields<OperationField>,
    pub nested: Flag,
    pub commutative: Flag,
    pub pure: Flag,
}

#[derive(FromField)]
pub struct OperationField {
    pub ident: Option<Ident>,
    pub ty: Type,
}
