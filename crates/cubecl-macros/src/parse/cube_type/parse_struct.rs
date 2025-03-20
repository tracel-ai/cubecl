use std::iter;

use darling::{FromDeriveInput, FromField, ast::Data, util::Flag};
use quote::format_ident;
use syn::{Generics, Ident, Type, Visibility, parse_quote, punctuated::Punctuated};

use crate::paths::prelude_type;

#[derive(FromDeriveInput, Debug)]
#[darling(supports(struct_named, struct_unit), attributes(expand, cube), map = unwrap_fields)]
pub struct CubeTypeStruct {
    pub ident: Ident,
    pub name_launch: Option<Ident>,
    pub name_comptime: Option<Ident>,
    pub name_expand: Option<Ident>,
    data: Data<(), TypeField>,
    #[darling(skip)]
    pub fields: Vec<TypeField>,
    pub generics: Generics,
    pub vis: Visibility,
}

#[derive(FromField, Clone, Debug)]
#[darling(attributes(expand, cube))]
pub struct TypeField {
    pub vis: Visibility,
    pub ident: Option<Ident>,
    pub ty: Type,
    pub comptime: Flag,
}

fn unwrap_fields(mut ty: CubeTypeStruct) -> CubeTypeStruct {
    // This will be supported inline with the next darling release
    let fields = ty.data.as_ref().take_struct().unwrap().fields;
    ty.fields = fields.into_iter().cloned().collect();

    let name = &ty.ident;
    ty.name_expand
        .get_or_insert_with(|| format_ident!("{name}Expand"));
    ty.name_launch
        .get_or_insert_with(|| format_ident!("{name}Launch"));
    ty.name_comptime
        .get_or_insert_with(|| format_ident!("{name}Comptime"));

    ty
}

impl CubeTypeStruct {
    pub fn expanded_generics(&self) -> Generics {
        let runtime = prelude_type("Runtime");
        let mut generics = self.generics.clone();
        generics.params.push(parse_quote![R: #runtime]);
        let all = iter::once(parse_quote!['a]).chain(generics.params);
        generics.params = Punctuated::from_iter(all);
        generics
    }

    pub fn assoc_generics(&self) -> Generics {
        let runtime = prelude_type("Runtime");
        parse_quote![<'a, R: #runtime>]
    }
}
