use darling::usage::{CollectTypeParams, GenericsExt, Purpose};
use proc_macro2::TokenStream;
use syn::{
    Generics, Ident, Token, WhereClause, parse_quote, punctuated::Punctuated, spanned::Spanned,
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

pub(crate) fn bounded_where_clause(
    generics: &Generics,
    runtime_fields: impl CollectTypeParams,
    bound: impl Fn(&Ident) -> TokenStream,
) -> Option<WhereClause> {
    let type_params = generics.declared_type_params();

    let used_params = runtime_fields.collect_type_params(&Purpose::BoundImpl.into(), &type_params);

    if used_params.is_empty() {
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

    for param in used_params {
        let bound = bound(param);
        where_clause.predicates.push(parse_quote!(#bound));
    }

    Some(where_clause)
}
