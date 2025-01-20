use darling::FromDeriveInput;
use proc_macro2::TokenStream;
use quote::quote;
use syn::DeriveInput;

use crate::parse::operation::{Operation, OperationVariant};

impl Operation {
    fn variants(&self) -> Vec<&OperationVariant> {
        self.data.as_ref().take_enum().unwrap()
    }

    fn generate_opcode_impl(&self) -> TokenStream {
        let opcode = &self.opcode_name;
        let has_children = self.with_children.is_present();
        let variants = self.variants();
        let match_variants = variants.iter().map(|variant| {
            let ident = &variant.ident;
            if has_children {
                quote![Self::#ident(child) => #opcode::#ident(cubecl_ir::OperationCore::op_code(child))]
            } else {
                quote![Self::#ident(_) => #opcode::#ident]
            }
        });
        quote! {
            match self {
                #(#match_variants),*
            }
        }
    }

    fn generate_args_impl(&self) -> TokenStream {
        let has_children = self.with_children.is_present();
        let variants = self.variants();
        let match_variants = variants.iter().map(|variant| {
            let ident = &variant.ident;
            if has_children {
                quote![Self::#ident(child) => cubecl_ir::OperationCore::args(child)]
            } else {
                quote![Self::#ident(args) => cubecl_ir::OperationArgs::into_args(args)]
            }
        });
        quote! {
            match self {
                #(#match_variants),*
            }
        }
    }

    fn generate_from_args_impl(&self) -> TokenStream {
        let opcode = &self.opcode_name;
        let has_children = self.with_children.is_present();
        let variants = self.variants();
        let match_variants = variants.iter().map(|variant| {
            let ident = &variant.ident;
            if has_children {
                quote![#opcode::#ident(child) => cubecl_ir::OperationCore::from_code_and_args(child, args)]
            } else {
                quote![#opcode::#ident => Self::#ident(cubecl_ir::OperationArgs::from_args(args))]
            }
        });
        quote! {
            match op_code {
                #(#match_variants),*
            }
        }
    }

    fn generate_operation_impl(&self) -> TokenStream {
        let name = &self.ident;
        let opcode_name = &self.opcode_name;
        let (generics, generic_names, where_clause) = self.generics.split_for_impl();

        let opcode_impl = self.generate_opcode_impl();
        let args_impl = self.generate_args_impl();
        let from_args_impl = self.generate_from_args_impl();

        quote![impl #generics cubecl_ir::OperationCore for #name #generic_names #where_clause {
            type OpCode = #opcode_name;

            fn op_code(&self) -> Self::OpCode {
                #opcode_impl
            }
            fn args(&self) -> SmallVec<[Variable; 4]> {
                #args_impl
            }
            fn from_code_and_args(op_code: Self::OpCode, args: &[Variable]) -> Option<Self> {
                #from_args_impl
            }
        }]
    }

    fn generate_opcode(&self) -> TokenStream {
        let vis = &self.vis;
        let name = &self.opcode_name;
        let has_children = self.with_children.is_present();
        let variants = self.variants();
        let variants = variants.iter().map(|variant| {
            let ident = &variant.ident;
            if has_children {
                let child_ty = &variant.fields.fields[0].ty;
                quote![#ident(<#child_ty as cubecl_ir::OperationCore>::OpCode)]
            } else {
                quote![#ident]
            }
        });
        quote! {
            #vis enum #name {
                #(#variants),*
            }
        }
    }
}

pub fn generate_operation(input: DeriveInput) -> syn::Result<TokenStream> {
    let operation = Operation::from_derive_input(&input)?;

    let opcode = operation.generate_opcode();
    let operation_impl = operation.generate_operation_impl();

    Ok(quote! {
        #opcode
        #operation_impl
    })
}
