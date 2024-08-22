use std::cell::RefCell;

use proc_macro2::TokenStream;
use quote::{format_ident, quote, quote_spanned, ToTokens};
use syn::{
    parse::Parse, spanned::Spanned, Attribute, FnArg, GenericParam, Generics, Ident, ItemFn, Meta,
    Pat, PatType, Receiver, Type, Visibility,
};

use crate::{ir_type, parse::kernel::Kernel, prefix_ir, scope::Context, statement::Statement};

impl ToTokens for Kernel {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let vis = &self.visibility;
        let name = &self.name;
        let generics = &self.generics;
        let global_vars = Context::default().current_scope().generate_vars();
        let statements = &self.statements;
        let return_type = &self.returns;
        let args = transform_args(&self.parameters);
        let statement_ty = prefix_ir(format_ident!("Statement"));
        let input_checks = self
            .parameters
            .iter()
            .map(|(_, ty, _)| {
                let span = ty.span();
                let check = prefix_ir(format_ident!("assert_valid_type"));
                quote_spanned! {span=>
                    #check::<#ty>();
                }
            })
            .collect::<Vec<_>>();
        let block = ir_type("Block");
        tokens.extend(quote! {
            #vis mod #name {
                use super::*;

                fn __check_inputs() {
                    #(#input_checks)*
                }

                #[allow(unused)]
                pub fn expand #generics(#(#args),*) -> #block<#return_type> {
                    #(#global_vars)*
                    {
                        let mut __statements = Vec::new();
                        #(#statements)*
                        #block::new(__statements)
                    }
                }
            }
        });
    }
}

fn transform_args(args: &[(Ident, Type, bool)]) -> Vec<TokenStream> {
    args.iter()
        .map(|(name, ty, is_const)| {
            let expr = ir_type("Expr");
            if *is_const {
                quote_spanned! {name.span()=>
                    #name: #ty
                }
            } else {
                quote_spanned! {name.span()=>
                    #name: impl #expr<Output = #ty> + 'static + Clone
                }
            }
        })
        .collect()
}
