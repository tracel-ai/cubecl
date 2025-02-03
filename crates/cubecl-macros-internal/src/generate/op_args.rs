use darling::FromDeriveInput;
use proc_macro2::TokenStream;
use quote::quote;
use syn::DeriveInput;

use crate::parse::op_args::OpArgs;

impl OpArgs {
    fn generate_into(&self) -> TokenStream {
        let mut tokens = quote![let mut args = alloc::vec::Vec::new();];
        for field in self.data.as_ref().take_struct().unwrap().fields {
            let ident = field.ident.as_ref().unwrap();
            tokens.extend(quote![args.extend(crate::FromArgList::as_arg_list(&self.#ident));]);
        }
        quote! {
            #tokens
            args
        }
    }

    fn generate_from(&self) -> TokenStream {
        let mut tokens = quote![];
        for field in self.data.as_ref().take_struct().unwrap().fields {
            let ident = field.ident.as_ref().unwrap();
            tokens.extend(quote![#ident: crate::FromArgList::from_arg_list(&mut args),]);
        }
        quote! {
            let mut args: alloc::collections::VecDeque<crate::Variable> = args.iter().cloned().collect();
            Self {
                #tokens
            }
        }
    }

    fn generate_args_impl(&self) -> TokenStream {
        let name = &self.ident;
        let (generics, generic_names, where_clause) = self.generics.split_for_impl();

        let into = self.generate_into();
        let from = self.generate_from();

        quote![impl #generics crate::OperationArgs for #name #generic_names #where_clause {
            fn from_args(args: &[Variable]) -> Option<Self> {
                Some({#from})
            }

            fn as_args(&self) -> Option<alloc::vec::Vec<crate::Variable>> {
                Some({#into})
            }
        }]
    }
}

pub fn generate_op_args(input: DeriveInput) -> syn::Result<TokenStream> {
    let args = OpArgs::from_derive_input(&input)?;
    Ok(args.generate_args_impl())
}
