use darling::usage::{GenericsExt, Purpose, UsesTypeParams};
use proc_macro2::TokenStream;
use syn::{
    Generics, Token, Type, WhereClause, parse_quote, punctuated::Punctuated, spanned::Spanned,
};

pub mod assign;
pub mod autotune;
pub mod cube_impl;
pub mod cube_trait;
pub mod cube_type;
pub mod expression;
pub mod into_runtime;
pub mod kernel;
pub mod launch;
pub mod statement;

pub trait RuntimeField: Clone + UsesTypeParams {
    fn ty(self) -> Type;
}

impl RuntimeField for syn::Field {
    fn ty(self) -> Type {
        self.ty
    }
}

pub(crate) fn bounded_where_clause<T: RuntimeField>(
    generics: &Generics,
    runtime_fields: impl IntoIterator<Item = T>,
    bound: impl Fn(&Type) -> TokenStream,
) -> Option<WhereClause> {
    let params = generics.declared_type_params();
    let opts = Purpose::BoundImpl.into();

    let fields_params = runtime_fields
        .into_iter()
        .filter(|it| !it.clone().uses_type_params(&opts, &params).is_empty())
        .map(|it| it.ty())
        .collect::<Vec<_>>();

    if fields_params.is_empty() {
        return generics.where_clause.clone();
    }

    let span = generics.span();
    let mut where_clause = generics
        .where_clause
        .clone()
        .unwrap_or_else(|| WhereClause {
            where_token: Token![where](span),
            predicates: Punctuated::new(),
        });

    for field_ty in fields_params {
        let bound = bound(&field_ty);
        where_clause.predicates.push(parse_quote!(#bound));
    }

    Some(where_clause)
}
