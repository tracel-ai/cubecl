//! Thin `proc-macro` shim: converts token streams and forwards to `cubecl-macros-core`, where the
//! expansion logic lives (so it can be tested and fuzzed outside a `proc-macro` crate).
#![allow(clippy::large_enum_variant)]

use proc_macro::TokenStream;
use quote::quote;

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
    cubecl_macros_core::cube(args.into(), input.into()).into()
}

/// Derive macro to define a cube type that is launched with a kernel
#[proc_macro_derive(CubeLaunch, attributes(cube, launch))]
pub fn module_derive_cube_launch(input: TokenStream) -> TokenStream {
    cubecl_macros_core::cube_type(input.into(), true).into()
}

/// Derive macro to define a cube type that is not launched
#[proc_macro_derive(CubeType, attributes(cube, expand))]
pub fn module_derive_cube_type(input: TokenStream) -> TokenStream {
    cubecl_macros_core::cube_type(input.into(), false).into()
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
    cubecl_macros_core::derive_expand(metadata.into(), input.into()).into()
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
#[proc_macro_derive(AutotuneKey, attributes(autotune))]
pub fn derive_autotune_key(input: TokenStream) -> TokenStream {
    cubecl_macros_core::autotune_key(input.into()).into()
}

/// Implements `IntoRuntime` for a `CubeType`
#[proc_macro_derive(IntoRuntime, attributes(cube))]
pub fn derive_into_runtime(input: TokenStream) -> TokenStream {
    cubecl_macros_core::into_runtime(input.into()).into()
}

/// Implements mutability for a `CubeType`
#[proc_macro_derive(CubeTypeMut, attributes(cube))]
pub fn derive_assign(input: TokenStream) -> TokenStream {
    cubecl_macros_core::cube_type_mut(input.into()).into()
}
