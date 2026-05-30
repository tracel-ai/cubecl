//! Fuzz `#[cube]` with random source text: text -> tokens -> `cube_impl`. The macro must return
//! `Ok`/`Err` for any input, never panic (a proc-macro panic aborts the user's compile). Empty
//! attribute args, so the item is fuzzed and `#[cube(debug)]`'s intentional panic isn't hit.
//!
//! The input is `&str`, so libfuzzer interprets each artifact as UTF-8 source: a crash file is
//! exactly the offending code (`cat` it, or `cargo fuzz fmt cube_macro <artifact>`).
#![no_main]

use libfuzzer_sys::fuzz_target;
use proc_macro2::TokenStream;

fuzz_target!(|text: &str| {
    if let Ok(tokens) = text.parse::<TokenStream>() {
        let _ = cubecl_macros_core::cube_impl(TokenStream::new(), tokens);
    }
});
