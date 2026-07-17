use proc_macro::TokenStream;
use syn::{DeriveInput, ItemImpl, parse_macro_input};
use type_hash::type_hash_impl;

use crate::{
    generate::{
        const_eval::generate_const_eval,
        cube_op::{generate_cube_op, generate_op_traits},
        pass_name::generate_pass_name,
        simplify::generate_simplify,
    },
    parse::{
        cube_op::{CubeOp, CubeOpArgs, PathList},
        from_meta_tokens,
    },
};

mod generate;
mod parse;
mod type_hash;

macro_rules! macro_try {
    ($op: expr) => {
        match $op {
            Ok(res) => res,
            Err(e) => return e.into_compile_error().into(),
        }
    };
}

#[proc_macro_derive(TypeHash, attributes(type_hash))]
pub fn derive_type_hash(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = syn::parse(input).unwrap();
    type_hash_impl(input).into()
}

/// Version of `pliron_op` that automatically derives shape ops, accessors, and a constructor from
/// struct fields. Also enables `format` by default, and sets the verifier to `"succ"` unless
/// overridden.
#[proc_macro_attribute]
pub fn cube_op(args: TokenStream, input: TokenStream) -> TokenStream {
    let args = macro_try!(from_meta_tokens(args.into()));
    let input = macro_try!(syn::parse(input));
    macro_try!(generate_cube_op(input, args)).into()
}

#[proc_macro_attribute]
pub fn op_traits(args: TokenStream, input: TokenStream) -> TokenStream {
    let traits = parse_macro_input!(args as PathList);
    let input = parse_macro_input!(input as DeriveInput);
    macro_try!(generate_op_traits(input, traits)).into()
}

/// Implements `Pass::name` on the annotated `impl Pass for X` block, returning the short name of
/// the pass struct (e.g. `"InsertEntrypointPass"`) instead of the fully-qualified type path.
#[proc_macro_attribute]
pub fn pass_name(_args: TokenStream, input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as ItemImpl);
    macro_try!(generate_pass_name(input)).into()
}

#[proc_macro]
pub fn const_eval(input: TokenStream) -> TokenStream {
    macro_try!(generate_const_eval(input.into())).into()
}

#[proc_macro]
pub fn simplify(input: TokenStream) -> TokenStream {
    macro_try!(generate_simplify(input.into())).into()
}
