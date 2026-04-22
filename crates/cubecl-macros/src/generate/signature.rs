use proc_macro2::TokenStream;
use quote::{ToTokens, quote};

use crate::{
    parse::{kernel::expand_kernel_ty, signature::*},
    paths::prelude_type,
};

impl ToTokens for KernelSignature {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let scope = prelude_type("Scope");

        let name = &self.name;
        let generics = &self.generics;
        let where_clause = &generics.where_clause;
        let scope_lifetime = &self.scope_lifetime;

        let return_type = match &self.returns {
            KernelReturns::ExpandType(ty) => {
                let normalized_ty = expand_kernel_ty(ty.clone(), false);
                quote![#normalized_ty]
            }
            KernelReturns::Plain(ty) => quote![#ty],
        };
        let out = if let Some(receiver) = &self.receiver_arg {
            let args = self.parameters.iter().skip(1);

            quote! {
                fn #name #generics(
                    #receiver,
                    scope: &#scope_lifetime #scope,
                    #(#args),*
                ) -> #return_type #where_clause
            }
        } else {
            let args = &self.parameters;
            quote! {
                fn #name #generics(
                    scope: &#scope_lifetime #scope,
                    #(#args),*
                ) -> #return_type #where_clause
            }
        };

        tokens.extend(out);
    }
}

impl ToTokens for KernelParam {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let name = &self.name;
        let ty = &self.normalized_ty;
        let mut_ = &self.mutability;
        tokens.extend(quote![#mut_ #name: #ty]);
    }
}
