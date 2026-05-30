// modified from https://github.com/elastio/bon/blob/master/bon-macros/src/error.rs

use proc_macro2::{TokenStream, TokenTree};
use quote::{ToTokens, quote};
use syn::parse::Parse;

use crate::parse::helpers::is_helper;

/// Handle the error returned from the macro logic. This may be either a syntax
/// error or a logic error. In either case, we want to return a [`TokenStream2`]
/// that still provides good IDE experience. See [`Fallback`] for details.
pub(crate) fn error_into_token_stream(err: syn::Error, item: TokenStream) -> TokenStream {
    let compile_error = err.to_compile_error();

    syn::parse2::<Fallback>(item)
        .map(|fallback| quote!(#compile_error #fallback))
        .unwrap_or_else(|_| compile_error)
}

/// This is used in error handling for better IDE experience. For example, while
/// the developer is writing the function code they'll have a bunch of syntax
/// errors in the process. While that happens the proc macro should output at
/// least some representation of the input code that the developer wrote with
/// a separate compile error entry. This keeps the syntax highlighting and IDE
/// type analysis, completions and other hints features working even if macro
/// fails to parse some syntax or finds some other logic errors.
///
/// This utility does very low-level parsing to strip helper attributes (i.e.
/// `#[comptime]`) from the input. This is to prevent the IDE from showing
/// errors for helper attributes that need to be processed by this macro.
struct Fallback {
    output: TokenStream,
}

impl Parse for Fallback {
    fn parse(input: syn::parse::ParseStream<'_>) -> syn::Result<Self> {
        let mut output = TokenStream::new();

        loop {
            let found_attr = input.step(|cursor| {
                let mut cursor = *cursor;
                while let Some((tt, next)) = cursor.token_tree() {
                    match &tt {
                        TokenTree::Group(group) => {
                            let fallback: Self = syn::parse2(group.stream())?;
                            let new_group =
                                proc_macro2::Group::new(group.delimiter(), fallback.output);

                            output.extend([TokenTree::Group(new_group)]);
                        }
                        TokenTree::Punct(punct) if punct.as_char() == '#' => {
                            return Ok((true, cursor));
                        }
                        TokenTree::Punct(_) | TokenTree::Ident(_) | TokenTree::Literal(_) => {
                            output.extend([tt]);
                        }
                    }

                    cursor = next;
                }

                Ok((false, cursor))
            })?;

            if !found_attr {
                return Ok(Self { output });
            }

            input
                .call(syn::Attribute::parse_outer)?
                .into_iter()
                .filter(|attr| !is_helper(attr))
                .for_each(|attr| attr.to_tokens(&mut output));
        }
    }
}

impl ToTokens for Fallback {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.output.to_tokens(tokens);
    }
}
