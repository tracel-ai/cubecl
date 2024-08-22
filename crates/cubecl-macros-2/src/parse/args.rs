use std::collections::HashSet;

use syn::{parse::Parse, punctuated::Punctuated, Ident, Token};

pub struct Args {
    /// This would hold launch, launch_unchecked
    pub options: HashSet<Ident>,
}

impl Parse for Args {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        // If more complex parsing is needed, it would go here.
        let acceptable_values = ["launch", "launch_unchecked"];
        let options: Result<HashSet<Ident>, _> =
            Punctuated::<Ident, Token![,]>::parse_terminated(input)?
                .into_iter()
                .map(|ident| {
                    if acceptable_values.contains(&ident.to_string().as_str()) {
                        Ok(ident)
                    } else {
                        Err(syn::Error::new_spanned(
                            ident,
                            "Only `launch` or `launch_unchecked` are allowed.",
                        ))
                    }
                })
                .collect();
        Ok(Args { options: options? })
    }
}
