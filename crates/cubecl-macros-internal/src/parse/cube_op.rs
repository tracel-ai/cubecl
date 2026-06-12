use darling::{FromDeriveInput, FromField, FromMeta, ast::Data, util::Flag};
use syn::{Attribute, Expr, Generics, Ident, LitStr, Type, Visibility};

#[derive(FromMeta)]
pub struct CubeOpArgs {
    pub name: LitStr,
    pub format: Option<LitStr>,
    #[darling(default)]
    pub verifier: Verifier,
}

#[derive(Default, FromMeta)]
pub enum Verifier {
    #[default]
    Succ,
    Custom,
}

#[derive(Default, FromMeta)]
pub enum ResultTy {
    #[default]
    None,
    SameAs(Ident),
    Fixed(Expr),
    FromInputs(Expr),
    Argument,
}

impl ResultTy {
    pub fn has_result(&self) -> bool {
        !matches!(self, ResultTy::None)
    }
}

#[derive(FromDeriveInput)]
#[darling(supports(struct_named), attributes(result_ty), forward_attrs)]
pub struct CubeOp {
    pub vis: Visibility,
    pub ident: Ident,
    pub generics: Generics,
    pub attrs: Vec<Attribute>,
    #[darling(with = unwrap_fields)]
    pub data: Vec<CubeOpArg>,
    #[darling(default, flatten)]
    pub result_ty: ResultTy,
}

#[derive(Clone)]
pub struct CubeOpArg {
    pub vis: Visibility,
    pub ident: syn::Ident,
    pub ty: syn::Type,
    pub kind: ArgKind,
    pub flags: ArgFlags,
}

#[derive(Clone, Copy)]
pub enum ArgKind {
    Value,
    Attribute,
}

impl From<CubeOpField> for CubeOpArg {
    fn from(value: CubeOpField) -> Self {
        let CubeOpField {
            vis,
            ident,
            ty,
            flags,
        } = value;
        let kind = ArgKind::from_type(&ty);
        Self {
            vis,
            ident: ident.unwrap(),
            ty,
            kind,
            flags,
        }
    }
}

impl ArgKind {
    pub fn from_type(ty: &Type) -> Self {
        let Type::Path(ty_path) = ty else {
            return ArgKind::Attribute;
        };
        let last_segment = ty_path.path.segments.last().unwrap();
        if last_segment.arguments.is_empty() && last_segment.ident == "Value" {
            ArgKind::Value
        } else {
            ArgKind::Attribute
        }
    }
}

#[derive(FromField)]
#[darling(forward_attrs, attributes(operand, attribute))]
struct CubeOpField {
    pub vis: Visibility,
    pub ident: Option<syn::Ident>,
    pub ty: syn::Type,
    #[darling(flatten)]
    pub flags: ArgFlags,
}

#[derive(FromMeta, Clone, Copy)]
pub struct ArgFlags {
    pub ptr_read: Flag,
    pub ptr_write: Flag,

    pub optional: Flag,
}

fn unwrap_fields(data: &syn::Data) -> darling::Result<Vec<CubeOpArg>> {
    let data = Data::<(), CubeOpField>::try_from(data)?;
    let fields = data.take_struct().unwrap().fields;
    let args = fields.into_iter().map(|it| it.into()).collect();
    Ok(args)
}
