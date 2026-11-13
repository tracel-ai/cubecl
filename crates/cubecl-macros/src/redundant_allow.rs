//! A warning on a lint a `#[cube]` function allows that the macro already allows on it, when the
//! `warn-redundant-allow` feature is on.

use proc_macro2::TokenStream;
use quote::quote;
use syn::Attribute;

/// The lints `#[cube]` allows on every function it expands.
#[cfg(feature = "warn-redundant-allow")]
const ALLOWED_BY_CUBE: [&str; 2] = ["dead_code", "clippy::too_many_arguments"];

/// One deprecation warning per lint in `attrs`' `#[allow(..)]` that `#[cube]` already allows,
/// spanned on that lint so it points at what to remove. A deprecation is the only warning a proc
/// macro can raise on stable.
#[cfg(feature = "warn-redundant-allow")]
pub fn warning(attrs: &[Attribute]) -> TokenStream {
    use syn::{Meta, Token, punctuated::Punctuated, spanned::Spanned};

    let warnings = attrs
        .iter()
        .filter(|attr| attr.path().is_ident("allow"))
        .filter_map(|attr| {
            attr.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
                .ok()
        })
        .flatten()
        .filter(|lint| {
            let path = lint.path().segments.iter();
            let name = path.map(|segment| segment.ident.to_string());
            ALLOWED_BY_CUBE.contains(&name.collect::<Vec<_>>().join("::").as_str())
        })
        .map(|lint| {
            let span = lint.span();
            quote::quote_spanned! {span=>
                const _: () = {
                    #[deprecated(note = "`#[cube]` already allows this lint on the function")]
                    #[allow(non_upper_case_globals)]
                    const redundant_allow: () = ();
                    redundant_allow
                };
            }
        });
    quote![#(#warnings)*]
}

#[cfg(not(feature = "warn-redundant-allow"))]
pub fn warning(_attrs: &[Attribute]) -> TokenStream {
    quote![]
}
