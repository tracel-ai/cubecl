use std::cell::RefCell;

use proc_macro2::TokenStream;
use quote::{format_ident, quote, quote_spanned, ToTokens};
use syn::{
    parse::Parse, spanned::Spanned, Attribute, FnArg, GenericParam, Generics, Ident, ItemFn, Meta,
    Pat, PatType, Receiver, Type, Visibility,
};

use crate::{
    ir_path, ir_type, parse::kernel::Kernel, prefix_ir, scope::Context, statement::Statement,
    IR_PATH,
};

impl ToTokens for Kernel {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let vis = &self.visibility;
        let name = &self.name;
        let generics = &self.generics;
        let global_vars = Context::default().current_scope().generate_vars();
        let block = &self.block;
        let return_type = &self.returns;
        let args = transform_args(&self.parameters);
        let statement_ty = prefix_ir(format_ident!("Statement"));
        let input_checks = self
            .parameters
            .iter()
            // Const can be anything as long as the accessed fields are cube types, since the access
            // gets resolved at expansion time and collapsed into a literal in the kernel
            .filter(|(_, _, is_const)| !is_const)
            .map(|(_, ty, _)| {
                let span = ty.span();
                let check = prefix_ir(format_ident!("assert_valid_type"));
                quote_spanned! {span=>
                    #check::<#ty>();
                }
            })
            .collect::<Vec<_>>();
        let expr = ir_type("Expr");
        let ir_path = ir_path();
        tokens.extend(quote! {
            #vis mod #name {
                use super::*;
                use #ir_path::{ExpandExpr as _, PartialExpand as _};

                fn __check_inputs() {
                    #(#input_checks)*
                }

                #[allow(unused, clippy::all)]
                pub fn expand #generics(#(#args),*) -> impl #expr<Output = #return_type> {
                    #(#global_vars)*
                    {
                        #block
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
