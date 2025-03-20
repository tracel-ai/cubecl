use darling::{FromDeriveInput, FromField, ast::Data};
use syn::{Generics, Ident};

#[derive(FromDeriveInput)]
#[darling(supports(struct_any))]
pub struct OpArgs {
    pub ident: Ident,
    pub generics: Generics,
    pub data: Data<(), OpArgsField>,
}

#[derive(FromField)]
#[darling(attributes(args))]
pub struct OpArgsField {
    pub ident: Option<Ident>,
}
