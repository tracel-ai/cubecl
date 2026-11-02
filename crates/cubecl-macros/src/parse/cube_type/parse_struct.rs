use darling::{FromDeriveInput, FromField, ast::Data, uses_type_params, util::Flag};
use quote::format_ident;
use syn::{Attribute, Generics, Ident, Meta, Token, Type, Visibility, punctuated::Punctuated};

use crate::{generate::RuntimeField, parse::harness::HarnessGate};

#[derive(FromDeriveInput, Debug)]
#[darling(
    supports(struct_named, struct_unit),
    attributes(expand, cube, launch),
    forward_attrs(must_use),
    map = unwrap_fields
)]
pub struct CubeTypeStruct {
    pub ident: Ident,
    /// Attributes the expand type carries too: a value that must be used must be in expansion.
    pub attrs: Vec<Attribute>,
    pub name_launch: Option<Ident>,
    pub name_comptime: Option<Ident>,
    pub name_expand: Option<Ident>,
    data: Data<(), TypeField>,
    #[darling(skip)]
    pub fields: Vec<TypeField>,
    pub generics: Generics,
    pub vis: Visibility,
    pub skip_bounds: Flag,
    pub derive: Option<syn::Meta>,
    /// The gate of the `CubeHarness` impls, by `cfg(test)` when absent.
    pub harness: Option<HarnessGate>,
    /// `#[repr(C)]`: the struct can be a buffer element (a record).
    #[darling(skip)]
    pub repr_c: bool,
}

/// Whether `attrs` declare `#[repr(C)]`. A record's fields are read at their `offset_of!`
/// offsets with natural alignment, so `packed` is an error.
pub fn parse_repr_c(attrs: &[Attribute]) -> syn::Result<bool> {
    let mut repr_c = false;
    for attr in attrs.iter().filter(|attr| attr.path().is_ident("repr")) {
        let metas = attr.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)?;
        for meta in metas {
            if meta.path().is_ident("C") {
                repr_c = true;
            } else if meta.path().is_ident("packed") {
                return Err(syn::Error::new_spanned(
                    meta,
                    "a `#[repr(C)]` cube type can't be `packed`: buffer record fields must be naturally aligned",
                ));
            }
        }
    }
    Ok(repr_c)
}

#[derive(FromField, Clone, Debug)]
#[darling(attributes(expand, cube))]
pub struct TypeField {
    pub vis: Visibility,
    pub ident: Option<Ident>,
    pub ty: Type,
    pub comptime: Flag,
}

uses_type_params!(TypeField, ty);
impl RuntimeField for TypeField {
    fn ty(self) -> Type {
        self.ty
    }
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
        self.generics.clone()
    }
}
