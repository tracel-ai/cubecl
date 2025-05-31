use generate::{
    op_args::generate_op_args,
    operation::{generate_opcode, generate_operation},
};
use proc_macro::TokenStream;
use type_hash::type_hash_impl;

mod generate;
mod parse;
mod type_hash;

/// *Internal macro*
///
/// Generates an implementation of `OperationArgs` for this type. All fields must implement
/// `FromArgList`.
#[doc(hidden)]
#[proc_macro_derive(OperationArgs, attributes(args))]
pub fn derive_operation_args(input: TokenStream) -> TokenStream {
    let input = syn::parse(input).unwrap();
    match generate_op_args(input) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.into_compile_error().into(),
    }
}

/// Generates reflection info for an operation. Generates an opcode enum and an implementation of
/// `OperationReflect` that deconstructs and reconstructs the typed version. All variant fields must
/// implement `OperationArgs`, or `OperationReflect` if the variant is nested. Uses the `operation`
/// helper attribute.
///
/// # Arguments
///
/// * `opcode_name` - the name of the generated opcode enum (required)
/// * `pure` - marks this entire operation as pure
/// * `commutative` - marks this entire operation as commutative
///
/// # Variant arguments
///
/// * `pure` - Marks this variant as pure
/// * `commutative` - Marks this variant as commutative
///
#[doc(hidden)]
#[proc_macro_derive(OperationReflect, attributes(operation))]
pub fn derive_operation(input: TokenStream) -> TokenStream {
    let input = syn::parse(input).unwrap();
    match generate_operation(input) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.into_compile_error().into(),
    }
}

/// Generates an opcode enum for an operation, without implementation `OperationReflect`. Allows for
/// manual implementation.
///
/// Use `self.__match_opcode()` to get the opcode for an operation.
///
/// # Arguments
///
/// * `opcode_name` - the name of the generated opcode enum (required)
#[doc(hidden)]
#[proc_macro_derive(OperationCode, attributes(operation))]
pub fn derive_opcode(input: TokenStream) -> TokenStream {
    let input = syn::parse(input).unwrap();
    match generate_opcode(input) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.into_compile_error().into(),
    }
}

#[proc_macro_derive(TypeHash, attributes(type_hash))]
pub fn derive_type_hash(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = syn::parse(input).unwrap();
    type_hash_impl(input).into()
}
