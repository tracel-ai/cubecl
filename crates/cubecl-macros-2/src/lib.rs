use darling::FromDeriveInput;
use error::error_into_token_stream;
use parse::{
    expand::Expand,
    expand_impl::ExpandImplVisitor,
    helpers::RemoveHelpers,
    kernel::{Kernel, KernelArgs},
};
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, visit_mut::VisitMut, DeriveInput, ItemFn, ItemImpl};

mod error;
mod expression;
mod generate;
mod parse;
mod scope;
mod statement;

mod paths {
    use proc_macro2::Span;
    use quote::format_ident;
    use std::cell::LazyCell;
    use syn::{Ident, Path, Token};

    #[allow(clippy::declare_interior_mutable_const)]
    const CORE_PATH: LazyCell<Path> = LazyCell::new(|| {
        let span = Span::call_site();
        let mut path = Path::from(format_ident!("cubecl_core"));
        path.leading_colon = Some(Token![::](span));
        path
    });
    #[allow(clippy::declare_interior_mutable_const)]
    const IR_PATH: LazyCell<Path> = LazyCell::new(|| {
        let mut path = core_path();
        path.segments.push(format_ident!("new_ir").into());
        path
    });
    #[allow(clippy::declare_interior_mutable_const)]
    const PRELUDE_PATH: LazyCell<Path> = LazyCell::new(|| {
        let mut path = core_path();
        path.segments.push(format_ident!("prelude").into());
        path
    });

    pub fn ir_path() -> Path {
        #[allow(clippy::borrow_interior_mutable_const)]
        IR_PATH.clone()
    }

    pub fn prelude_path() -> Path {
        #[allow(clippy::borrow_interior_mutable_const)]
        PRELUDE_PATH.clone()
    }

    pub fn core_path() -> Path {
        #[allow(clippy::borrow_interior_mutable_const)]
        CORE_PATH.clone()
    }

    pub fn prefix_ir(ident: Ident) -> Path {
        let mut path = ir_path();
        path.segments.push(ident.into());
        path
    }

    pub fn core_type(ty: &str) -> Path {
        let mut path = core_path();
        let ident = format_ident!("{ty}");
        path.segments.push(ident.into());
        path
    }

    pub fn ir_type(ty: &str) -> Path {
        let mut path = ir_path();
        let ident = format_ident!("{ty}");
        path.segments.push(ident.into());
        path
    }

    pub fn prelude_type(ty: &str) -> Path {
        let mut path = prelude_path();
        let ident = format_ident!("{ty}");
        path.segments.push(ident.into());
        path
    }
}
pub(crate) use paths::{core_type, ir_path, ir_type, prefix_ir, prelude_type};

#[proc_macro_attribute]
pub fn cube2(args: TokenStream, input: TokenStream) -> TokenStream {
    match cube2_impl(args, input.clone()) {
        Ok(tokens) => tokens,
        Err(e) => error_into_token_stream(e, input.into()).into(),
    }
}

fn cube2_impl(args: TokenStream, input: TokenStream) -> syn::Result<TokenStream> {
    let args = KernelArgs::from_tokens(args.into())?;
    let mut function: ItemFn = syn::parse(input)?;
    let kernel = Kernel::from_item_fn(function.clone(), args)?;
    RemoveHelpers.visit_item_fn_mut(&mut function);

    Ok(TokenStream::from(quote! {
        #function
        #kernel
    }))
}

#[proc_macro_derive(Expand, attributes(expand))]
pub fn derive_square_type(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let expand = match Expand::from_derive_input(&input) {
        Ok(expand) => expand,
        Err(e) => return e.write_errors().into(),
    };
    quote![#expand].into()
}

#[proc_macro_attribute]
pub fn expand_impl(_args: TokenStream, input: TokenStream) -> TokenStream {
    let mut impl_block = parse_macro_input!(input as ItemImpl);
    let mut visitor = ExpandImplVisitor::default();
    visitor.visit_item_impl_mut(&mut impl_block);
    let expansion = visitor.0.unwrap();

    TokenStream::from(quote! {
        #impl_block
        #expansion
    })
}
