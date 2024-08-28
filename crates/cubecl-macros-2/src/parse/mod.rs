use syn::{visit_mut::VisitMut, GenericParam, TypeParam};

pub mod branch;
pub mod expand;
pub mod expand_impl;
pub mod expression;
pub mod helpers;
pub mod kernel;
pub mod operator;

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
