#![allow(unused)]

use std::{cell::LazyCell, collections::HashSet};

use generate::strip_comptime;
use parse::{args::Args, kernel::Kernel, kernel_struct::KernelStruct};
use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{format_ident, quote};
use statement::Statement;
use syn::{
    parse::Parse, parse_macro_input, punctuated::Punctuated, Ident, ItemFn, Path, PathSegment,
    Token,
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
    let in_2 = input.clone();
    let kernel = parse_macro_input!(in_2 as Kernel);
    let mut function = parse_macro_input!(input as ItemFn);
    strip_comptime(&mut function);

    TokenStream::from(quote! {
        #function
        #kernel
    })
}

#[proc_macro_derive(KernelArg)]
pub fn derive_square_type(input: TokenStream) -> TokenStream {
    let kernel_struct = parse_macro_input!(input as KernelStruct);

    TokenStream::from(quote![#kernel_struct])
}
