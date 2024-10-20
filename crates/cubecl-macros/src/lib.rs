use core::panic;

use darling::FromDeriveInput;
use error::error_into_token_stream;
use generate::autotune::{generate_autotune_key, generate_autotune_set};
use parse::{
    cube_impl::CubeImpl,
    cube_trait::{CubeTrait, CubeTraitImpl},
    cube_type::CubeType,
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
                let mut expand_impl = CubeTraitImpl::from_item_impl(item_impl)?;
                let expand_impl = expand_impl.to_tokens_mut();

                Ok(TokenStream::from(quote! {
                    #expand_impl
                }))
            } else {
                let mut expand_impl = CubeImpl::from_item_impl(item_impl)?;
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
#[proc_macro_derive(CubeLaunch, attributes(expand))]
pub fn module_derive_cube_launch(input: TokenStream) -> TokenStream {
    gen_cube_type(input, true)
}

/// Derive macro to define a cube type that is not launched
#[proc_macro_derive(CubeType, attributes(expand))]
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

/// Implements display and initialization for autotune keys.
///
/// # Helper
///
/// Use the `#[autotune]` helper attribute to anchor fields to the next power of two, or rename
/// the fields for the display implementation.
///
/// # Example
/// ```ignore
/// #[derive(AutotuneKey, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
/// pub struct OperationKey {
///     #[autotune(name = "Batch Size")]
///     batch_size: usize,
///     channels: usize,
///     #[autotune(anchor(max = 1024))]
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

/// Crates a tuning set with a specific signature. Should return a tuple of benchmark inputs.
///
/// # Arguments
///
/// * `name` - the name of the generated operations struct (default: `PascalCaseFnName`)
/// * `key` - the name of the key input parameter (default: `key`)
/// * `create_key` - path to function that creates the key. If not specified, `new` must be implemented manually.
/// * `should_run` - path to override function for the `should_run` function of the set.
/// * `operations` - ordered list of operations returned by this tune set
///
/// # Example
///
/// ```ignore
/// #[tune(create_key = key_from_input, operations(operation_1, operation_2))]
/// pub fn my_operations(key: MyKey, input: JitTensor<f32, 4>) -> JitTensor<f32, 4> {
///     let bench_input = random_tensor_like(input, -1.0, 1.0);
///     
///     (bench_input)
/// }
///
/// fn key_from_input(input: &JitTensor<f32, 4>) -> MyKey {
///     MyKey::new(input.shape.dims)
/// }
/// ```
#[proc_macro_attribute]
pub fn tune(args: TokenStream, input: TokenStream) -> TokenStream {
    match autotune_set_impl(args, input.clone()) {
        Ok(tokens) => tokens,
        Err(e) => error_into_token_stream(e, input.into()).into(),
    }
}

fn autotune_set_impl(args: TokenStream, input: TokenStream) -> syn::Result<TokenStream> {
    let item = syn::parse(input)?;
    let args = from_tokens(args.into())?;
    Ok(generate_autotune_set(item, args)?.into())
}
