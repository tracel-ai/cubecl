use syn::{GenericParam, TypeParam, visit_mut::VisitMut};

pub mod autotune;
pub mod branch;
pub mod cube_impl;
pub mod cube_trait;
pub mod cube_type;
pub mod desugar;
pub mod expression;
pub mod helpers;
pub mod kernel;
pub mod operator;
pub mod statement;

pub struct StripDefault;
impl VisitMut for StripDefault {
    fn visit_generics_mut(&mut self, i: &mut syn::Generics) {
        for generic in i.params.iter_mut() {
            match generic {
                GenericParam::Lifetime(_) => {}
                GenericParam::Type(ty) => {
                    ty.default.take();
                    ty.eq_token.take();
                }
                GenericParam::Const(con) => {
                    con.default.take();
                    con.eq_token.take();
                }
            }
        }
    }
}

pub struct StripBounds;

impl VisitMut for StripBounds {
    fn visit_generics_mut(&mut self, i: &mut syn::Generics) {
        for generic in i.params.iter_mut() {
            match generic {
                GenericParam::Lifetime(lifetime) => {
                    lifetime.attrs.clear();
                    lifetime.bounds.clear();
                    lifetime.colon_token.take();
                }
                GenericParam::Type(ty) => {
                    ty.attrs.clear();
                    ty.bounds.clear();
                    ty.colon_token.take();
                }
                GenericParam::Const(con) => {
                    *generic = GenericParam::Type(TypeParam {
                        attrs: Default::default(),
                        ident: con.ident.clone(),
                        colon_token: None,
                        bounds: Default::default(),
                        eq_token: None,
                        default: None,
                    })
                }
            }
        }
    }
}
