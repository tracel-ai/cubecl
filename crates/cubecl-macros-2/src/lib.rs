#![allow(unused)]

use std::{cell::LazyCell, collections::HashSet};

use parse::{
    args::Args,
    expand_impl::ExpandImplVisitor,
    helpers::RemoveHelpers,
    kernel::Kernel,
    kernel_struct::{FieldExpand, MethodExpand},
};
use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{format_ident, quote};
use statement::Statement;
use syn::{
    parse::Parse, parse_macro_input, punctuated::Punctuated, visit_mut::VisitMut, Ident, ItemFn,
    ItemImpl, Path, PathSegment, Token,
};

mod expression;
mod generate;
mod parse;
mod scope;
mod statement;

const IR_PREFIX: &'static str = "::cubecl_core::new_ir::";
const IR_PATH: LazyCell<Path> = LazyCell::new(|| {
    let span = Span::call_site();
    let mut path = Path::from(format_ident!("cubecl_core"));
    path.segments.push(format_ident!("new_ir").into());
    path.leading_colon = Some(Token![::](span));
    path
});

pub(crate) fn prefix_ir(ident: Ident) -> Path {
    let mut path = IR_PATH.clone();
    path.segments.push(ident.into());
    path
}
pub(crate) fn ir_type(ty: &str) -> Path {
    let ident = format_ident!("{ty}");
    let mut path = IR_PATH.clone();
    path.segments.push(ident.into());
    path
}

#[proc_macro_attribute]
pub fn cube2(args: TokenStream, input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args as Args);
    let mut function = parse_macro_input!(input as ItemFn);
    let kernel = match Kernel::from_item_fn(function.clone()) {
        Ok(kernel) => kernel,
        Err(e) => return TokenStream::from(e.to_compile_error()),
    };
    RemoveHelpers.visit_item_fn_mut(&mut function);

    TokenStream::from(quote! {
        #function
        #kernel
    })
}

#[proc_macro_derive(KernelArg)]
pub fn derive_square_type(input: TokenStream) -> TokenStream {
    let kernel_struct = parse_macro_input!(input as FieldExpand);

    TokenStream::from(quote![#kernel_struct])
}

#[proc_macro_derive(CubeMethods)]
pub fn derive_cube_methods(input: TokenStream) -> TokenStream {
    let cube_methods = parse_macro_input!(input as MethodExpand);

    TokenStream::from(quote![#cube_methods])
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
