//! Fuzz `#[cube]` with random source text: text -> tokens -> `cube_impl`. The macro must return
//! `Ok`/`Err` for any input, never panic (a proc-macro panic aborts the user's compile). Empty
//! attribute args, so the item is fuzzed and `#[cube(debug)]`'s intentional panic isn't hit.
//!
//! The input is `&str`, so libfuzzer interprets each artifact as UTF-8 source: a crash file is
//! exactly the offending code (`cat` it, or `cargo fuzz fmt cube_macro <artifact>`).
//!
//! Deeply nested input is skipped first: `syn` parses some constructs (notably the qualified-path
//! `<Type as Trait>::x` reached through a run of `<`) by recursing once per token with no depth
//! limit, so such input overflows the stack inside `syn::parse2` before `cube_impl` runs. A stack
//! overflow aborts the process and cannot be caught, so it would just wedge the fuzzer on a `syn`
//! limitation rather than a `cube` bug. Real kernels nest far below `MAX_NESTING`.
#![no_main]

use libfuzzer_sys::fuzz_target;
use proc_macro2::{TokenStream, TokenTree};

/// Recursion-depth ceiling handed to `syn`. Chosen well below the observed stack-overflow
/// threshold (~1000 levels under the sanitizer build) with margin to spare.
const MAX_NESTING: usize = 256;

/// Conservative over-estimate of `syn`'s parse recursion depth for a token stream: group nesting
/// plus the longest run of consecutive punctuation (a run of `<`, `&`, `*`, `-`, ... each recurses
/// once per token). Never under-estimates, so it is a safe gate. Its own recursion is bounded by
/// `MAX_NESTING` because it stops descending once the cap is exceeded.
fn token_nesting(tokens: &TokenStream) -> usize {
    fn walk(tokens: &TokenStream, base: usize, max: &mut usize) {
        let mut run = 0usize;
        for tt in tokens.clone() {
            match tt {
                TokenTree::Group(group) => {
                    run = 0;
                    let depth = base + 1;
                    *max = (*max).max(depth);
                    if depth <= MAX_NESTING {
                        walk(&group.stream(), depth, max);
                    }
                }
                TokenTree::Punct(_) => {
                    run += 1;
                    *max = (*max).max(base + run);
                }
                _ => run = 0,
            }
        }
    }
    let mut max = 0;
    walk(tokens, 0, &mut max);
    max
}

fuzz_target!(|text: &str| {
    if let Ok(tokens) = text.parse::<TokenStream>() {
        if token_nesting(&tokens) > MAX_NESTING {
            return;
        }
        let _ = cubecl_macros_core::cube_impl(TokenStream::new(), tokens);
    }
});
