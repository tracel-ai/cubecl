#![allow(unused)]

use std::{cell::LazyCell, collections::HashSet};

use error::error_into_token_stream;
use parse::{
    args::Args, expand_impl::ExpandImplVisitor, helpers::RemoveHelpers, kernel::Kernel,
    kernel_struct::Expand,
};
use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{format_ident, quote};
use statement::Statement;
use syn::{
    parse::Parse, parse_macro_input, punctuated::Punctuated, visit_mut::VisitMut, Ident, ItemFn,
    ItemImpl, Path, PathSegment, Token,
};

mod error;
mod expression;
mod generate;
mod parse;
mod scope;
mod statement;

const IR_PREFIX: &str = "::cubecl_core::new_ir::";
#[allow(clippy::declare_interior_mutable_const)]
const IR_PATH: LazyCell<Path> = LazyCell::new(|| {
    let span = Span::call_site();
    let mut path = Path::from(format_ident!("cubecl_core"));
    path.segments.push(format_ident!("new_ir").into());
    path.leading_colon = Some(Token![::](span));
    path
});

pub(crate) fn ir_path() -> Path {
    #[allow(clippy::borrow_interior_mutable_const)]
    IR_PATH.clone()
}

pub(crate) fn prefix_ir(ident: Ident) -> Path {
    let mut path = ir_path();
    path.segments.push(ident.into());
    path
}
pub(crate) fn ir_type(ty: &str) -> Path {
    let mut path = ir_path();
    let ident = format_ident!("{ty}");
    path.segments.push(ident.into());
    path
}

#[proc_macro_attribute]
pub fn cube2(args: TokenStream, input: TokenStream) -> TokenStream {
    match cube2_impl(args, input.clone()) {
        Ok(tokens) => tokens,
        Err(e) => error_into_token_stream(e, input.into()).into(),
    }
}

fn cube2_impl(args: TokenStream, input: TokenStream) -> syn::Result<TokenStream> {
    let args: Args = syn::parse(args)?;
    let mut function: ItemFn = syn::parse(input)?;
    let kernel = Kernel::from_item_fn(function.clone())?;
    RemoveHelpers.visit_item_fn_mut(&mut function);

    Ok(TokenStream::from(quote! {
        #function
        #kernel
    }))
}

#[proc_macro_derive(Expand)]
pub fn derive_square_type(input: TokenStream) -> TokenStream {
    let kernel_struct = parse_macro_input!(input as Expand);

    TokenStream::from(quote![#kernel_struct])
}

#[proc_macro_attribute]
pub fn expand_impl(args: TokenStream, input: TokenStream) -> TokenStream {
    let mut impl_block = parse_macro_input!(input as ItemImpl);
    let mut visitor = ExpandImplVisitor::default();
    visitor.visit_item_impl_mut(&mut impl_block);
    let expansion = visitor.0.unwrap();

    TokenStream::from(quote! {
        #impl_block
        #expansion
    })
}
