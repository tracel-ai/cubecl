use darling::{ast::Data, FromDeriveInput, FromField};
use quote::format_ident;
use syn::{visit_mut::VisitMut, Expr, Generics, Ident, Type, Visibility};

use super::{StripBounds, StripDefault};

#[derive(FromDeriveInput)]
#[darling(supports(struct_any), attributes(expand), and_then = unwrap_fields)]
pub struct Expand {
    pub vis: Visibility,
    pub generics: Generics,
    #[darling(skip)]
    pub generic_names: Generics,
    pub ident: Ident,
    #[darling(default)]
    pub name: Option<Ident>,
    #[darling(default)]
    pub ir_type: Option<Expr>,
    data: Data<(), ExpandField>,
    #[darling(skip)]
    pub fields: Vec<ExpandField>,
}

fn unwrap_fields(mut expand: Expand) -> darling::Result<Expand> {
    let fields = expand.data.as_ref().take_struct().unwrap().fields;
    let fields = fields.into_iter().cloned().enumerate();
    expand.fields = fields
        .filter(|(_, field)| !is_phantom_data(&field.ty) && !field.skip)
        .map(|(i, mut field)| {
            field.name = field
                .ident
                .as_ref()
                .map(|it| it.to_string())
                .unwrap_or_else(|| i.to_string());
            field
        })
        .collect();
    expand
        .name
        .get_or_insert_with(|| format_ident!("{}Expand", expand.ident));
    StripDefault.visit_generics_mut(&mut expand.generics);
    expand.generic_names = expand.generics.clone();
    StripBounds.visit_generics_mut(&mut expand.generic_names);
    Ok(expand)
}

#[derive(FromField, Clone)]
#[darling(attributes(expand))]
pub struct ExpandField {
    pub vis: Visibility,
    pub ident: Option<Ident>,
    #[darling(skip)]
    pub name: String,
    pub ty: Type,
    #[darling(default)]
    pub skip: bool,
}

fn is_phantom_data(field: &Type) -> bool {
    match &field {
        Type::Path(path) => {
            let last = path.path.segments.last().unwrap();
            last.ident == "PhantomData"
        }
        _ => false,
    }
}
