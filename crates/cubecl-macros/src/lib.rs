#![allow(clippy::large_enum_variant)]

use core::panic;

use darling::util::Flag;
use error::error_into_token_stream;
use generate::autotune::generate_autotune_key;
use parse::{
    cube_impl::CubeImpl,
    cube_trait::{CubeTrait, CubeTraitImpl},
    helpers::RemoveHelpers,
    kernel::{KernelArgs, Launch, from_tokens},
};
use proc_macro::TokenStream;
use quote::quote;
use syn::{Attribute, Item, ItemFn, Meta, visit_mut::VisitMut};

use crate::{
    generate::{
        asm::generate_asm_unexpanded, assign::generate_cube_type_mut,
        into_runtime::generate_into_runtime,
    },
    parse::{
        cube_type::{generate_cube_harness, generate_cube_type},
        derive_expand::generate_derive_expand,
        helpers::ReplaceDefines,
    },
};

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
/// * `create_dummy_kernel` - Generates a function to create a kernel without launching it. Used for
///   testing.
///
/// # Trait arguments
/// * `expand_base_traits` - base traits for the expanded "second half" of a trait with methods.
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
    let mut args: KernelArgs = from_tokens(args.into())?;

    // `#[cube_harness]` below `#[cube]` hasn't run yet, so take it from here.
    if let Item::Fn(func) = &mut item {
        let len = func.attrs.len();
        func.attrs.retain(|attr| !is_attr(attr, "cube_harness"));
        if func.attrs.len() != len {
            args.harness = Flag::present();
        }
    }

    expand_cube(args, item)
}

fn is_attr(attr: &Attribute, name: &str) -> bool {
    attr.path()
        .segments
        .last()
        .is_some_and(|seg| seg.ident == name)
}

fn expand_cube(args: KernelArgs, mut item: Item) -> syn::Result<TokenStream> {
    let tokens = match item.clone() {
        Item::Fn(kernel) => {
            let kernel = Launch::from_item_fn(kernel, args)?;
            RemoveHelpers.visit_item_mut(&mut item);
            ReplaceDefines.visit_item_mut(&mut item);

            let extra_allow = match kernel.func.context.is_intrinsic {
                true => quote![#[allow(unused_variables)]],
                false => quote![],
            };

            return Ok(TokenStream::from(quote! {
                #[allow(dead_code, clippy::too_many_arguments)]
                #extra_allow
                #item
                #kernel
            }));
        }
        Item::Trait(kernel_trait) => {
            let is_debug = args.debug.is_present();
            let expand_trait = CubeTrait::from_item_trait(kernel_trait, args)?;

            let tokens = TokenStream::from(quote! {
                #expand_trait
            });
            if is_debug {
                panic!("{tokens}");
            }
            return Ok(tokens);
        }
        Item::Impl(item_impl) => {
            if item_impl.trait_.is_some() {
                let mut expand_impl = CubeTraitImpl::from_item_impl(item_impl, &args)?;
                let expand_impl = expand_impl.to_tokens_mut();

                Ok(TokenStream::from(quote! {
                    #expand_impl
                }))
            } else {
                let mut expand_impl = CubeImpl::from_item_impl(item_impl, &args)?;
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

/// Generate a `harness` function for a cube function, which runs it on a
/// single unit of a runtime and hands its result back to the host, so a test
/// can call `f::harness::<R>(..)` instead of writing a kernel, buffers and a
/// read back by hand.
///
/// Goes above or below `#[cube]`, and is usually gated so harnesses only exist
/// in tests, with `#[cfg_attr(test, cube_harness)]`. Every parameter must be a
/// scalar, a slice or array of scalars, a comptime scalar or a cube type that
/// derives `CubeHarness`, and the function must return them or a tuple of them,
/// or nothing when it writes into a mutable slice.
///
/// # Example
///
/// ```ignored
/// #[cfg_attr(test, cube_harness)]
/// #[cube]
/// fn add<F: Float>(x: F, y: F) -> F {
///     x + y
/// }
///
/// #[test]
/// fn adds() {
///     assert_eq!(add::harness::<CpuRuntime, f32>(1.0, 2.0), 3.0);
/// }
/// ```
#[proc_macro_attribute]
pub fn cube_harness(args: TokenStream, input: TokenStream) -> TokenStream {
    match cube_harness_impl(args, input.clone()) {
        Ok(tokens) => tokens,
        Err(e) => error_into_token_stream(e, input.into()).into(),
    }
}

fn cube_harness_impl(args: TokenStream, input: TokenStream) -> syn::Result<TokenStream> {
    if !args.is_empty() {
        return Err(syn::Error::new(
            proc_macro2::TokenStream::from(args)
                .into_iter()
                .next()
                .unwrap()
                .span(),
            "`#[cube_harness]` takes no arguments",
        ));
    }
    let mut item: ItemFn = syn::parse(input)?;

    // Expand the `#[cube]` below in place, so the harness lands in the module
    // it generates for the function.
    let cube = item
        .attrs
        .iter()
        .position(|attr| is_attr(attr, "cube"))
        .ok_or_else(|| {
            syn::Error::new_spanned(
                &item.sig.ident,
                "`#[cube_harness]` only goes on a `#[cube]` function",
            )
        })?;
    let cube_args = match item.attrs.remove(cube).meta {
        Meta::Path(_) => proc_macro2::TokenStream::new(),
        Meta::List(list) => list.tokens,
        meta @ Meta::NameValue(_) => {
            return Err(syn::Error::new_spanned(
                meta,
                "expected `#[cube]` or `#[cube(..)]`",
            ));
        }
    };
    let mut args: KernelArgs = from_tokens(cube_args)?;
    args.harness = Flag::present();

    expand_cube(args, Item::Fn(item))
}

/// Derive macro to define a cube type that is launched with a kernel
#[proc_macro_derive(CubeLaunch, attributes(cube, launch))]
pub fn module_derive_cube_launch(input: TokenStream) -> TokenStream {
    gen_cube_type(input, true)
}

/// Derive macro to define a cube type that is not launched
#[proc_macro_derive(CubeType, attributes(cube, expand))]
pub fn module_derive_cube_type(input: TokenStream) -> TokenStream {
    gen_cube_type(input, false)
}

/// Derive macro to let a cube type cross between the host and a cube
/// function's `harness`, as an input or an output. The type must also derive
/// `CubeType` and `CubeLaunch`.
///
/// Like `#[cube_harness]`, it can be gated to tests with
/// `#[cfg_attr(test, derive(CubeHarness))]`.
#[proc_macro_derive(CubeHarness, attributes(cube, launch))]
pub fn module_derive_cube_harness(input: TokenStream) -> TokenStream {
    let parsed = syn::parse(input);

    let input = match &parsed {
        Ok(val) => val,
        Err(err) => return err.to_compile_error().into(),
    };

    match generate_cube_harness(input) {
        Ok(val) => val.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn gen_cube_type(input: TokenStream, with_launch: bool) -> TokenStream {
    let parsed = syn::parse(input);

    let input = match &parsed {
        Ok(val) => val,
        Err(err) => return err.to_compile_error().into(),
    };

    match generate_cube_type(input, with_launch) {
        Ok(val) => val.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// Attribute macro to define a type that can be used as a kernel comptime
/// argument This derive Debug, Hash, `PartialEq`, Eq, Clone, Copy
#[proc_macro_attribute]
pub fn derive_cube_comptime(_metadata: TokenStream, input: TokenStream) -> TokenStream {
    let input: proc_macro2::TokenStream = input.into();
    quote! {
        #[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
        #input
    }
    .into()
}

/// Attribute macro to derive cube traits for existing structs, without redefining that struct.
#[proc_macro_attribute]
pub fn derive_expand(metadata: TokenStream, input: TokenStream) -> TokenStream {
    match generate_derive_expand(input.into(), metadata.into()) {
        Ok(val) => val.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// Mark the contents of this macro as compile time values, turning off all
/// expansion for this code and using it verbatim
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

/// Mark the contents of this macro as an intrinsic, turning off all expansion
/// for this code and calling it with the scope
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

/// GPU version of [`asm`](std::arch::asm). Currently parses all the same options, but most are not
/// applicable to GPU assembly architectures. Should validate and give proper errors at some point.
/// Also adds a new register spec: the inferred register specifier (`_`). This is because the
/// specifier isn't actually meaningful in PTX and is currently ignored, with constraints being
/// inferred from the value type. The reason it's still present is because we may want to add an
/// explicit `mem` specifier to allow fine-grained memory clobbering, or support for other assembly
/// formats that do use different register types.
///
/// # Example
/// ```ignored
/// #use cubecl_macros::cube;
/// #[cube]
/// fn do_stuff(input: u32) -> u32 {
///     let mut out: u32;
///     gpu_asm!("some.custom.ptx {} {}", out(_) out, in(_) input)
/// }
/// ```
#[proc_macro]
pub fn gpu_asm(input: TokenStream) -> TokenStream {
    match generate_asm_unexpanded(input.into()) {
        Ok(val) => val.into(),
        Err(err) => err.to_compile_error().into(),
    }
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
/// TODO: calling a trait method returning `comptime_type` from
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
/// This terminates the execution of the unit even if nested inside many
/// functions.
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
/// For now, only an exponential function is supported, and it can be modified
/// with `exp`. By default, the base is '2' and there are no `min` or `max`
/// provided.
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

/// Implements `IntoRuntime` for a `CubeType`
#[proc_macro_derive(IntoRuntime, attributes(cube))]
pub fn derive_into_runtime(input: TokenStream) -> TokenStream {
    let input = syn::parse(input).unwrap();
    match generate_into_runtime(&input) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.into_compile_error().into(),
    }
}

/// Implements mutability for a `CubeType`
#[proc_macro_derive(CubeTypeMut, attributes(cube))]
pub fn derive_assign(input: TokenStream) -> TokenStream {
    let input = syn::parse(input).unwrap();
    match generate_cube_type_mut(&input) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.into_compile_error().into(),
    }
}
