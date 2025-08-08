#![allow(clippy::large_enum_variant)]

use core::panic;

use darling::FromDeriveInput;
use error::error_into_token_stream;
use generate::autotune::generate_autotune_key;
use parse::{
    cube_impl::CubeImpl,
    cube_trait::{CubeTrait, CubeTraitImpl},
    cube_type::CubeType,
    helpers::{RemoveHelpers, ReplaceIndices},
    kernel::{Launch, from_tokens},
};
use proc_macro::TokenStream;
use quote::quote;
use syn::{Item, visit_mut::VisitMut};

mod error;
mod expression;
mod generate;
mod operator;
mod parse;
mod paths;
mod scope;
mod statement;

/// Mark a cube function, trait or implementation for expansion.
///
/// # Arguments
/// * `launch` - generates a function to launch the kernel
/// * `launch_unchecked` - generates a launch function without checks
/// * `debug` - panics after generation to print the output to console
/// * `create_dummy_kernel` - Generates a function to create a kernel without launching it. Used for testing.
///
/// # Example
///
/// ```ignored
/// # use cubecl_macros::cube;
/// #[cube]
/// fn my_addition(a: u32, b: u32) -> u32 {
///     a + b
/// }
/// ```
#[proc_macro_attribute]
pub fn cube(args: TokenStream, input: TokenStream) -> TokenStream {
    match cube_impl(args, input.clone()) {
        Ok(tokens) => tokens,
        Err(e) => error_into_token_stream(e, input.into()).into(),
    }
}

fn cube_impl(args: TokenStream, input: TokenStream) -> syn::Result<TokenStream> {
    let mut item: Item = syn::parse(input)?;
    let args = from_tokens(args.into())?;

    let tokens = match item.clone() {
        Item::Fn(kernel) => {
            let kernel = Launch::from_item_fn(kernel, args)?;
            RemoveHelpers.visit_item_mut(&mut item);
            ReplaceIndices.visit_item_mut(&mut item);

            return Ok(TokenStream::from(quote! {
                #[allow(dead_code, clippy::too_many_arguments)]
                #item
                #kernel
            }));
        }
        Item::Trait(kernel_trait) => {
            let expand_trait = CubeTrait::from_item_trait(kernel_trait)?;

            Ok(TokenStream::from(quote! {
                #expand_trait
            }))
        }
        Item::Impl(item_impl) => {
            if item_impl.trait_.is_some() {
                let mut expand_impl = CubeTraitImpl::from_item_impl(
                    item_impl,
                    args.src_file,
                    args.debug_symbols.is_present(),
                )?;
                let expand_impl = expand_impl.to_tokens_mut();

                Ok(TokenStream::from(quote! {
                    #expand_impl
                }))
            } else {
                let mut expand_impl = CubeImpl::from_item_impl(
                    item_impl,
                    args.src_file,
                    args.debug_symbols.is_present(),
                )?;
                let expand_impl = expand_impl.to_tokens_mut();

                Ok(TokenStream::from(quote! {
                    #expand_impl
                }))
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

/// Derive macro to define a cube type that is launched with a kernel
#[proc_macro_derive(CubeLaunch, attributes(expand, cube))]
pub fn module_derive_cube_launch(input: TokenStream) -> TokenStream {
    gen_cube_type(input, true)
}

/// Derive macro to define a cube type that is not launched
#[proc_macro_derive(CubeType, attributes(expand, cube))]
pub fn module_derive_cube_type(input: TokenStream) -> TokenStream {
    gen_cube_type(input, false)
}

fn gen_cube_type(input: TokenStream, with_launch: bool) -> TokenStream {
    let parsed = syn::parse(input);

    let input = match &parsed {
        Ok(val) => val,
        Err(err) => return err.to_compile_error().into(),
    };

    let cube_type = match CubeType::from_derive_input(input) {
        Ok(val) => val,
        Err(err) => return err.write_errors().into(),
    };

    cube_type.generate(with_launch).into()
}

/// Attribute macro to define a type that can be used as a kernel comptime argument
/// This derive Debug, Hash, PartialEq, Eq, Clone, Copy
#[proc_macro_attribute]
pub fn derive_cube_comptime(_metadata: TokenStream, input: TokenStream) -> TokenStream {
    let input: proc_macro2::TokenStream = input.into();
    quote! {
        #[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
        #input
    }
    .into()
}

/// Mark the contents of this macro as compile time values, turning off all expansion for this code
/// and using it verbatim
///
/// # Example
/// ```ignored
/// #use cubecl_macros::cube;
/// #fn some_rust_function(a: u32) -> u32 {}
/// #[cube]
/// fn do_stuff(input: u32) -> u32 {
///     let comptime_value = comptime! { some_rust_function(3) };
///     input + comptime_value
/// }
/// ```
#[proc_macro]
pub fn comptime(input: TokenStream) -> TokenStream {
    let tokens: proc_macro2::TokenStream = input.into();
    quote![{ #tokens }].into()
}

/// Mark the contents of this macro as an intrinsic, turning off all expansion for this code
/// and calling it with the scope
///
/// # Example
/// ```ignored
/// #use cubecl_macros::cube;
/// #[cube]
/// fn do_stuff(input: u32) -> u32 {
///     let comptime_value = intrinsic! { |scope| u32::elem_size(scope) };
///     input + comptime_value
/// }
/// ```
#[proc_macro]
pub fn intrinsic(_input: TokenStream) -> TokenStream {
    quote![{ cubecl::unexpanded!() }].into()
}

/// Makes the function return a compile time value
/// Useful in a cube trait to have a part of the trait return comptime values
///
/// # Example
/// ```ignored
/// #use cubecl_macros::cube;
/// #[cube]
/// fn do_stuff(#[comptime] input: u32) -> comptime_type!(u32) {
///     input + 5   
/// }
/// ```
///
/// TODO: calling a trait method returning comptime_type from
/// within another trait method does not work
#[proc_macro]
pub fn comptime_type(input: TokenStream) -> TokenStream {
    let tokens: proc_macro2::TokenStream = input.into();
    quote![ #tokens ].into()
}

/// Insert a literal comment into the kernel source code.
///
/// # Example
/// ```ignored
/// #use cubecl_macros::cube;
/// #[cube]
/// fn do_stuff(input: u32) -> u32 {
///     comment!("Add five to the input");
///     input + 5
/// }
/// ```
#[proc_macro]
pub fn comment(input: TokenStream) -> TokenStream {
    let tokens: proc_macro2::TokenStream = input.into();
    quote![{ #tokens }].into()
}

/// Terminate the execution of the kernel for the current unit.
///
/// This terminates the execution of the unit even if nested inside many functions.
///
/// # Example
/// ```ignored
/// #use cubecl_macros::cube;
/// #[cube]
/// fn stop_if_more_than_ten(input: u32)  {
///     if input > 10 {
///         terminate!();
///     }
/// }
/// ```
#[proc_macro]
pub fn terminate(input: TokenStream) -> TokenStream {
    let tokens: proc_macro2::TokenStream = input.into();
    quote![{ #tokens }].into()
}

/// Implements display and initialization for autotune keys.
///
/// # Helper
///
/// Use the `#[autotune(anchor)]` helper attribute to anchor a numerical value.
/// This groups multiple numerical values into the same bucket.
///
/// For now, only an exponential function is supported, and it can be modified with `exp`.
/// By default, the base is '2' and there are no `min` or `max` provided.
///
/// # Example
/// ```ignore
/// #[derive(AutotuneKey, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
/// pub struct OperationKey {
///     #[autotune(name = "Batch Size")]
///     batch_size: usize,
///     channels: usize,
///     #[autotune(anchor(exp(min = 16, max = 1024, base = 2)))]
///     height: usize,
///     #[autotune(anchor)]
///     width: usize,
/// }
/// ```
#[proc_macro_derive(AutotuneKey, attributes(autotune))]
pub fn derive_autotune_key(input: TokenStream) -> TokenStream {
    let input = syn::parse(input).unwrap();
    match generate_autotune_key(input) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.into_compile_error().into(),
    }
}
