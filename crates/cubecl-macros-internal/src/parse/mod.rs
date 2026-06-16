use darling::{FromMeta, ast::NestedMeta};
use proc_macro2::TokenStream;

pub mod const_eval;
pub mod cube_op;

pub fn from_meta_tokens<T: FromMeta>(tokens: TokenStream) -> syn::Result<T> {
    let meta = NestedMeta::parse_meta_list(tokens)?;
    T::from_list(&meta).map_err(syn::Error::from)
}
