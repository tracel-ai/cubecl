//! `#[cube]` expansion logic, kept in a normal library (not the `cubecl-macros` proc-macro crate)
//! so it can be tested and fuzzed.

#![allow(clippy::large_enum_variant)]

mod error;
mod expression;
mod generate;
mod operator;
mod parse;
mod paths;
mod scope;
mod statement;

use error::error_into_token_stream;
use generate::{
    assign::generate_cube_type_mut, autotune::generate_autotune_key,
    into_runtime::generate_into_runtime,
};
use parse::{
    cube_impl::CubeImpl,
    cube_trait::{CubeTrait, CubeTraitImpl},
    cube_type::generate_cube_type,
    derive_expand::generate_derive_expand,
    helpers::{RemoveHelpers, ReplaceDefines},
    kernel::{Launch, from_tokens},
};
use proc_macro2::TokenStream;
use quote::quote;
use syn::{Item, visit_mut::VisitMut};

/// Expand a `#[cube]` item, returning a `compile_error!` stream on failure.
pub fn cube(args: TokenStream, input: TokenStream) -> TokenStream {
    match cube_impl(args, input.clone()) {
        Ok(tokens) => tokens,
        Err(e) => error_into_token_stream(e, input),
    }
}

/// Fallible core of `#[cube]` expansion, and the fuzz entry point: must return `Ok`/`Err`, never panic.
pub fn cube_impl(args: TokenStream, input: TokenStream) -> syn::Result<TokenStream> {
    let mut item: Item = syn::parse2(input)?;
    let args = from_tokens(args)?;

    let tokens = match item.clone() {
        Item::Fn(kernel) => {
            let kernel = Launch::from_item_fn(kernel, args)?;
            RemoveHelpers.visit_item_mut(&mut item);
            ReplaceDefines.visit_item_mut(&mut item);

            let extra_allow = match kernel.func.context.is_intrinsic {
                true => quote![#[allow(unused_variables)]],
                false => quote![],
            };

            return Ok(quote! {
                #[allow(dead_code, clippy::too_many_arguments)]
                #extra_allow
                #item
                #kernel
            });
        }
        Item::Trait(kernel_trait) => {
            let is_debug = args.debug.is_present();
            let expand_trait = CubeTrait::from_item_trait(kernel_trait, args)?;

            let tokens = quote! {
                #expand_trait
            };
            if is_debug {
                panic!("{tokens}");
            }
            return Ok(tokens);
        }
        Item::Impl(item_impl) => {
            if item_impl.trait_.is_some() {
                let mut expand_impl = CubeTraitImpl::from_item_impl(item_impl, &args)?;
                let expand_impl = expand_impl.to_tokens_mut();

                Ok(quote! {
                    #expand_impl
                })
            } else {
                let mut expand_impl = CubeImpl::from_item_impl(item_impl, &args)?;
                let expand_impl = expand_impl.to_tokens_mut();

                Ok(quote! {
                    #expand_impl
                })
            }
        }
        item => Err(syn::Error::new_spanned(
            item,
            "`#[cube]` is only supported on traits and functions",
        ))?,
    };

    if args.debug.is_present() {
        match tokens {
            Ok(tokens) => panic!("{tokens}"),
            Err(err) => panic!("{err}"),
        };
    }

    tokens
}

/// Expand the `CubeLaunch` / `CubeType` derives.
pub fn cube_type(input: TokenStream, with_launch: bool) -> TokenStream {
    let parsed = syn::parse2(input);

    let input = match &parsed {
        Ok(val) => val,
        Err(err) => return err.to_compile_error(),
    };

    match generate_cube_type(input, with_launch) {
        Ok(val) => val,
        Err(err) => err.to_compile_error(),
    }
}

/// Expand the `derive_expand` attribute.
pub fn derive_expand(metadata: TokenStream, input: TokenStream) -> TokenStream {
    match generate_derive_expand(input, metadata) {
        Ok(val) => val,
        Err(err) => err.to_compile_error(),
    }
}

/// Expand the `AutotuneKey` derive.
pub fn autotune_key(input: TokenStream) -> TokenStream {
    let input = syn::parse2(input).unwrap();
    match generate_autotune_key(input) {
        Ok(tokens) => tokens,
        Err(e) => e.into_compile_error(),
    }
}

/// Expand the `IntoRuntime` derive.
pub fn into_runtime(input: TokenStream) -> TokenStream {
    let input = syn::parse2(input).unwrap();
    match generate_into_runtime(&input) {
        Ok(tokens) => tokens,
        Err(e) => e.into_compile_error(),
    }
}

/// Expand the `CubeTypeMut` derive.
pub fn cube_type_mut(input: TokenStream) -> TokenStream {
    let input = syn::parse2(input).unwrap();
    match generate_cube_type_mut(&input) {
        Ok(tokens) => tokens,
        Err(e) => e.into_compile_error(),
    }
}
