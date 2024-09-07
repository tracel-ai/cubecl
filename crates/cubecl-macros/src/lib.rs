use error::error_into_token_stream;
use generate::cube_type::generate_cube_type;
use parse::{
    cube_trait::{CubeTrait, CubeTraitImpl},
    helpers::{RemoveHelpers, ReplaceIndices},
    kernel::{from_tokens, Launch},
};
use proc_macro::TokenStream;
use quote::quote;
use syn::{visit_mut::VisitMut, Item};

mod error;
mod expression;
mod generate;
mod operator;
mod parse;
mod paths;
mod scope;
mod statement;

#[proc_macro_attribute]
pub fn cube(args: TokenStream, input: TokenStream) -> TokenStream {
    match cube_impl(args, input.clone()) {
        Ok(tokens) => tokens,
        Err(e) => error_into_token_stream(e, input.into()).into(),
    }
}

fn cube_impl(args: TokenStream, input: TokenStream) -> syn::Result<TokenStream> {
    let mut item: Item = syn::parse(input)?;
    match item.clone() {
        Item::Fn(kernel) => {
            let args = from_tokens(args.into())?;
            let kernel = Launch::from_item_fn(kernel, args)?;
            RemoveHelpers.visit_item_mut(&mut item);
            ReplaceIndices.visit_item_mut(&mut item);

            Ok(TokenStream::from(quote! {
                #[allow(dead_code, clippy::too_many_arguments)]
                #item
                #kernel
            }))
        }
        Item::Trait(kernel_trait) => {
            let expand_trait = CubeTrait::from_item_trait(kernel_trait)?;

            Ok(TokenStream::from(quote! {
                #expand_trait
            }))
        }
        Item::Impl(item_impl) if item_impl.trait_.is_some() => {
            let mut expand_impl = CubeTraitImpl::from_item_impl(item_impl)?;
            let expand_impl = expand_impl.to_tokens_mut();

            Ok(TokenStream::from(quote! {
                #expand_impl
            }))
        }
        item => Err(syn::Error::new_spanned(
            item,
            "`#[cube]` is only supported on traits and functions",
        ))?,
    }
}

// Derive macro to define a cube type that is launched with a kernel
#[proc_macro_derive(CubeLaunch, attributes(expand))]
pub fn module_derive_cube_launch(input: TokenStream) -> TokenStream {
    let input = syn::parse(input).unwrap();

    generate_cube_type(&input, true).into()
}

// Derive macro to define a cube type that is not launched
#[proc_macro_derive(CubeType, attributes(expand))]
pub fn module_derive_cube_type(input: TokenStream) -> TokenStream {
    let input = syn::parse(input).unwrap();

    generate_cube_type(&input, false).into()
}
