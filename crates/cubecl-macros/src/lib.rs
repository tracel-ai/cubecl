use darling::FromDeriveInput;
use error::error_into_token_stream;
use generate::cube_type::generate_cube_type;
use parse::{
    cube_trait::{CubeTrait, CubeTraitImpl},
    expand::{Expand, StaticExpand},
    expand_impl::ExpandImplVisitor,
    helpers::RemoveHelpers,
    kernel::{from_tokens, Launch},
};
use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use syn::{parse_macro_input, visit_mut::VisitMut, DeriveInput, Item, ItemImpl};

mod error;
mod expression;
mod generate;
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

            Ok(TokenStream::from(quote! {
                #[allow(dead_code, clippy::too_many_arguments)]
                #item
                #kernel
            }))
        }
        Item::Trait(kernel_trait) => {
            let args = from_tokens(args.into())?;
            let expand_trait = CubeTrait::from_item_trait(kernel_trait, args)?;

            Ok(TokenStream::from(quote! {
                #expand_trait
            }))
        }
        Item::Impl(item_impl) if item_impl.trait_.is_some() => {
            let args = from_tokens(args.into())?;
            let expand_impl = CubeTraitImpl::from_item_impl(item_impl, args)?;
            RemoveHelpers.visit_item_mut(&mut item);

            Ok(TokenStream::from(quote! {
                #[allow(dead_code, clippy::too_many_arguments)]
                #item
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
#[proc_macro_derive(CubeLaunch, attributes(cube_type))]
pub fn module_derive_cube_launch(input: TokenStream) -> TokenStream {
    let input = syn::parse(input).unwrap();

    generate_cube_type(&input, true).into()
}

// Derive macro to define a cube type that is not launched
#[proc_macro_derive(CubeType, attributes(cube_type))]
pub fn module_derive_cube_type(input: TokenStream) -> TokenStream {
    let input = syn::parse(input).unwrap();

    generate_cube_type(&input, false).into()
}

#[proc_macro_derive(Expand, attributes(expand))]
pub fn derive_expand(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let expand = match Expand::from_derive_input(&input) {
        Ok(expand) => expand,
        Err(e) => return e.write_errors().into(),
    };
    expand.to_token_stream().into()
}

// #[proc_macro_derive(CubeType, attributes(expand))]
// pub fn derive_cube_type(input: TokenStream) -> TokenStream {
//     let input = parse_macro_input!(input as DeriveInput);
//     let expand = match Runtime::from_derive_input(&input) {
//         Ok(expand) => expand,
//         Err(e) => return e.write_errors().into(),
//     };
//     expand.to_token_stream().into()
// }

#[proc_macro_derive(StaticExpand, attributes(expand))]
pub fn derive_static_expand(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let expand = match StaticExpand::from_derive_input(&input) {
        Ok(expand) => expand,
        Err(e) => return e.write_errors().into(),
    };
    expand.to_token_stream().into()
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
