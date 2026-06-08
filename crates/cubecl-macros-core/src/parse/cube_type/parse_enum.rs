use darling::{FromDeriveInput, util::Flag};
use proc_macro2::Span;
use quote::format_ident;
use syn::{Expr, ExprLit, Generics, Ident, Lit, parse_quote, spanned::Spanned};

use crate::paths::prelude_type;

#[derive(Debug)]
pub struct CubeTypeEnum {
    pub ident: Ident,
    pub name_expand: Ident,
    pub variants: Vec<CubeTypeVariant>,
    pub generics: syn::Generics,
    pub vis: syn::Visibility,
    pub runtime_variants: bool,
    pub with_constructors: bool,
    pub skip_bounds: bool,
    pub derive: Option<syn::Meta>,
}

#[derive(Debug, FromDeriveInput)]
#[darling(attributes(cube, launch, expand), supports(enum_any))]
pub struct CubeTypeEnumRepr {
    ident: Ident,
    vis: syn::Visibility,
    generics: syn::Generics,
    data: darling::ast::Data<syn::Variant, ()>,
    runtime_variants: Flag,
    /// Don't generate constructors, useful for expanding existing types where a new impl isn't allowed
    no_constructors: Flag,
    pub skip_bounds: Flag,
    derive: Option<syn::Meta>,
}

#[derive(Debug)]
pub struct CubeTypeVariant {
    pub ident: Ident,
    pub fields: syn::Fields,
    pub field_names: Vec<Ident>,
    pub kind: VariantKind,
    pub discriminant: i32,
}

#[derive(Debug)]
pub enum VariantKind {
    Named,
    Unnamed,
    Empty,
}

impl FromDeriveInput for CubeTypeEnum {
    fn from_derive_input(input: &syn::DeriveInput) -> darling::Result<Self> {
        let repr = CubeTypeEnumRepr::from_derive_input(input)?;
        let mut next_discriminant = 0;
        match &repr.data {
            darling::ast::Data::Enum(variants) => Ok(Self {
                name_expand: format_ident!("{}Expand", repr.ident),
                ident: repr.ident,
                generics: repr.generics,
                vis: repr.vis,
                runtime_variants: repr.runtime_variants.is_present(),
                with_constructors: !repr.no_constructors.is_present(),
                skip_bounds: repr.skip_bounds.is_present(),
                derive: repr.derive,
                variants: variants
                    .iter()
                    .map(|a| -> Result<_, syn::Error> {
                        let mut kind = if a.fields.is_empty() {
                            VariantKind::Empty
                        } else {
                            VariantKind::Unnamed
                        };

                        for field in a.fields.iter() {
                            if field.ident.is_some() {
                                kind = VariantKind::Named;
                            }
                        }

                        let discriminant = if let Some((_, disc)) = &a.discriminant {
                            let value = parse_discriminant(disc)?;
                            next_discriminant = value + 1;
                            value
                        } else {
                            let value = next_discriminant;
                            next_discriminant += 1;
                            value
                        };

                        Ok(CubeTypeVariant {
                            kind,
                            ident: a.ident.clone(),
                            field_names: a
                                .fields
                                .iter()
                                .enumerate()
                                .map(|(i, field)| match &field.ident {
                                    Some(name) => name.clone(),
                                    None => {
                                        Ident::new(format!("arg_{i}").as_str(), Span::call_site())
                                    }
                                })
                                .collect(),
                            fields: a.fields.clone(),
                            discriminant,
                        })
                    })
                    .collect::<Result<_, _>>()?,
            }),
            _ => Err(darling::Error::custom("Only enum are supported.")),
        }
    }
}

impl CubeTypeEnum {
    pub fn expanded_generics(&self) -> Generics {
        let runtime = prelude_type("Runtime");
        let mut generics = self.generics.clone();
        if !self.is_empty() {
            generics.params.push(parse_quote![R: #runtime]);
        }
        generics
    }

    pub fn assoc_generics(&self) -> Generics {
        let runtime = prelude_type("Runtime");
        parse_quote![<R: #runtime>]
    }

    pub fn is_empty(&self) -> bool {
        self.variants
            .iter()
            .all(|it| matches!(it.kind, VariantKind::Empty))
    }
}

// Discriminants can't be const expressions, but do allow basic arithmetic (and negate is required
// for negative discriminants)
fn parse_discriminant(expr: &Expr) -> syn::Result<i32> {
    let span = expr.span();
    match expr {
        Expr::Group(group) => parse_discriminant(&group.expr),
        Expr::Paren(paren) => parse_discriminant(&paren.expr),
        Expr::Cast(cast) => parse_discriminant(&cast.expr),
        Expr::Binary(binary) => {
            let lhs = parse_discriminant(&binary.left)?;
            let rhs = parse_discriminant(&binary.right)?;
            match binary.op {
                syn::BinOp::Add(_) => Ok(lhs + rhs),
                syn::BinOp::Sub(_) => Ok(lhs - rhs),
                syn::BinOp::Mul(_) => Ok(lhs * rhs),
                syn::BinOp::Div(_) => Ok(lhs / rhs),
                syn::BinOp::Rem(_) => Ok(lhs % rhs),
                syn::BinOp::BitXor(_) => Ok(lhs ^ rhs),
                syn::BinOp::BitAnd(_) => Ok(lhs & rhs),
                syn::BinOp::BitOr(_) => Ok(lhs | rhs),
                syn::BinOp::Shl(_) => Ok(lhs << rhs),
                syn::BinOp::Shr(_) => Ok(lhs >> rhs),
                _ => Err(syn::Error::new(span, "Unknown binary op in discriminant")),
            }
        }
        Expr::Lit(ExprLit {
            lit: Lit::Int(int), ..
        }) => int.base10_parse(),
        Expr::Unary(unary) => {
            let inner = parse_discriminant(&unary.expr)?;
            match unary.op {
                syn::UnOp::Not(_) => Ok(!inner),
                syn::UnOp::Neg(_) => Ok(-inner),
                _ => Err(syn::Error::new(span, "Unknown unary op in discriminant")),
            }
        }
        _ => Err(syn::Error::new(span, "Unsupported discriminant value")),
    }
}
