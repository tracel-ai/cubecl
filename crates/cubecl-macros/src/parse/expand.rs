use darling::{ast::Data, util::Flag, FromDeriveInput, FromField};
use quote::format_ident;
use syn::{visit_mut::VisitMut, Expr, Generics, Ident, Type, Visibility};

use super::StripDefault;

#[derive(FromDeriveInput)]
#[darling(supports(struct_any), attributes(expand), and_then = unwrap_fields)]
pub struct Expand {
    pub vis: Visibility,
    pub generics: Generics,
    pub ident: Ident,
    #[darling(default)]
    pub name: Option<Ident>,
    #[darling(default)]
    pub ir_type: Option<Expr>,
    data: Data<(), ExpandField>,
    #[darling(skip)]
    pub fields: Vec<ExpandField>,
}

#[derive(FromDeriveInput)]
#[darling(supports(struct_any), attributes(expand), and_then = unwrap_fields_static)]
pub struct StaticExpand {
    pub vis: Visibility,
    pub generics: Generics,
    pub ident: Ident,
    #[darling(default)]
    pub name: Option<Ident>,
}

#[derive(FromDeriveInput)]
#[darling(supports(struct_named), attributes(runtime), and_then = unwrap_runtime)]
pub struct Runtime {
    pub vis: Visibility,
    pub generics: Generics,
    pub ident: Ident,
    #[darling(default)]
    pub name: Option<Ident>,
    #[darling(default)]
    pub ir_type: Option<Expr>,
    data: Data<(), RuntimeField>,
    #[darling(skip)]
    pub fields: Vec<RuntimeField>,
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
    StripDefault.visit_generics_mut(&mut expand.generics);
    Ok(expand)
}

fn unwrap_runtime(mut runtime: Runtime) -> darling::Result<Runtime> {
    let fields = runtime.data.as_ref().take_struct().unwrap();
    runtime.fields = fields.into_iter().cloned().collect();
    runtime
        .fields
        .sort_by_key(|field| field.ident.as_ref().unwrap().to_string());
    StripDefault.visit_generics_mut(&mut runtime.generics);
    Ok(runtime)
}

fn unwrap_fields_static(mut expand: StaticExpand) -> darling::Result<StaticExpand> {
    expand
        .name
        .get_or_insert_with(|| format_ident!("{}Expand", expand.ident));
    StripDefault.visit_generics_mut(&mut expand.generics);
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
    pub comptime: Flag,
}

#[derive(FromField, Clone)]
#[darling(attributes(expand))]
pub struct RuntimeField {
    pub vis: Visibility,
    pub ident: Option<Ident>,
    pub ty: Type,
    pub comptime: Flag,
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
