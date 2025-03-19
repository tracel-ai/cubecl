use darling::{FromDeriveInput, util::Flag};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::DeriveInput;

use crate::parse::operation::{OpCode, Operation, OperationVariant};

impl Operation {
    fn variants(&self) -> Vec<&OperationVariant> {
        self.data.as_ref().take_enum().unwrap()
    }

    fn generate_opcode_impl(&self) -> TokenStream {
        let opcode = &self.opcode_name;
        let variants = self.variants();
        let match_variants = variants.iter().map(|variant| {
            let ident = &variant.ident;
            if variant.nested.is_present() {
                quote![Self::#ident(child) => #opcode::#ident(crate::OperationReflect::op_code(child))]
            } else if variant.fields.is_empty() {
                quote![Self::#ident => #opcode::#ident]
            } else if variant.fields.fields[0].ident.is_some() {
                quote![Self::#ident { .. } => #opcode::#ident]
            } else {
                let args = variant.fields.fields.iter().map(|_| quote![_]);
                quote![Self::#ident(#(#args),*) => #opcode::#ident]
            }
        });
        quote! {
            match self {
                #(#match_variants),*
            }
        }
    }

    fn generate_args_impl(&self) -> TokenStream {
        let variants = self.variants();
        let match_variants = variants.iter().map(|variant| {
            let ident = &variant.ident;
            if variant.nested.is_present() {
                quote![Self::#ident(child) => crate::OperationReflect::args(child)]
            } else if variant.fields.is_empty() {
                quote![Self::#ident => Some(alloc::vec::Vec::new())]
            } else if variant.fields.fields[0].ident.is_some() {
                let names = variant
                    .fields
                    .fields
                    .iter()
                    .map(|it| it.ident.clone().unwrap())
                    .collect::<Vec<_>>();
                let body = quote![{
                    let mut args = alloc::vec::Vec::new();
                    #(args.extend(crate::FromArgList::as_arg_list(#names));)*
                    Some(args)
                }];
                quote![Self::#ident { #(#names),* } => #body]
            } else {
                quote![Self::#ident(args) => crate::OperationArgs::as_args(args)]
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
        let variants = self.variants();
        let match_variants = variants.iter().map(|variant| {
            let ident = &variant.ident;
            if variant.nested.is_present() {
                quote![#opcode::#ident(child) => Some(Self::#ident(crate::OperationReflect::from_code_and_args(child, args)?))]
            } else if variant.fields.is_empty() {
                quote![#opcode::#ident => Some(Self::#ident)]
            } else if variant.fields.fields[0].ident.is_some() {
                let fields = variant
                    .fields
                    .fields
                    .iter()
                    .map(|it| it.ident.clone().unwrap())
                    .map(|it| quote![#it: crate::FromArgList::from_arg_list(&mut args)]);
                quote![#opcode::#ident => {
                    let mut args: alloc::collections::VecDeque<crate::Variable> = args.iter().cloned().collect();
                    Some(Self::#ident {
                        #(#fields),*
                    })
                }]
            } else {
                quote![#opcode::#ident => crate::OperationArgs::from_args(args).map(Self::#ident)]
            }
        });
        quote! {
            match op_code {
                #(#match_variants),*
            }
        }
    }

    fn generate_bool_property(
        &self,
        flag_global: impl Fn(&Self) -> Flag,
        flag_local: impl Fn(&OperationVariant) -> Flag,
        func_ident: &str,
    ) -> TokenStream {
        let func_ident = format_ident!("{func_ident}");
        if flag_global(self).is_present() {
            quote! {
                fn #func_ident(&self) -> bool { true }
            }
        } else {
            let variants = self.variants();
            let variants = variants.iter().map(|variant| {
                let ident = &variant.ident;
                let value = flag_local(variant).is_present();
                if variant.nested.is_present() && !value {
                    quote![Self::#ident(child) => crate::OperationReflect::#func_ident(child)]
                } else if variant.fields.is_empty() {
                    quote![Self::#ident => #value]
                } else if variant.fields.is_struct() {
                    quote![Self::#ident { .. } => #value]
                } else {
                    let args = variant.fields.fields.iter().map(|_| quote![_]);
                    quote![Self::#ident(#(#args),*) => #value]
                }
            });
            quote! {
                fn #func_ident(&self) -> bool {
                    match self {
                        #(#variants),*
                    }
                }
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
        let commutative =
            self.generate_bool_property(|x| x.commutative, |x| x.commutative, "is_commutative");
        let pure = self.generate_bool_property(|x| x.pure, |x| x.pure, "is_pure");

        quote![impl #generics crate::OperationReflect for #name #generic_names #where_clause {
            type OpCode = #opcode_name;

            fn op_code(&self) -> Self::OpCode {
                #opcode_impl
            }
            fn args(&self) -> Option<alloc::vec::Vec<crate::Variable>> {
                #args_impl
            }
            fn from_code_and_args(op_code: Self::OpCode, args: &[crate::Variable]) -> Option<Self> {
                #from_args_impl
            }

            #commutative
            #pure
        }]
    }

    fn generate_opcode(&self) -> TokenStream {
        let vis = &self.vis;
        let name = &self.opcode_name;
        let variants = self.variants();
        let variants = variants.iter().map(|variant| {
            let ident = &variant.ident;
            if variant.nested.is_present() {
                let child_ty = &variant.fields.fields[0].ty;
                quote![#ident(<#child_ty as crate::OperationReflect>::OpCode)]
            } else {
                quote![#ident]
            }
        });
        quote! {
            #[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord, derive_more::From)]
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

pub fn generate_opcode(input: DeriveInput) -> syn::Result<TokenStream> {
    let operation = OpCode::from_derive_input(&input)?;
    let operation = Operation {
        ident: operation.ident,
        vis: operation.vis,
        generics: operation.generics,
        data: operation.data,
        opcode_name: operation.opcode_name,
        commutative: Flag::default(),
        pure: Flag::default(),
    };

    let name = &operation.ident;
    let generics = &operation.generics;
    let opcode_name = &operation.opcode_name;
    let opcode = operation.generate_opcode();
    let match_opcode = operation.generate_opcode_impl();

    Ok(quote! {
        #opcode

        impl #generics #name {
            fn __match_opcode(&self) -> #opcode_name {
                #match_opcode
            }
        }
    })
}
