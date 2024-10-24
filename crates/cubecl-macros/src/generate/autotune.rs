use darling::FromDeriveInput;
use ident_case::RenameRule;
use proc_macro2::TokenStream;
use quote::{format_ident, quote, ToTokens};
use syn::{DeriveInput, Ident, Index, ItemFn, Member, Path};

use crate::{
    parse::autotune::{
        operation_name, AutotuneKey, AutotuneKeyField, AutotuneOperations, AutotuneOperationsArgs,
    },
    paths::tune_type,
};

impl AutotuneKey {
    fn generate_fmt_str(&self) -> String {
        let name = self.ident.to_string();
        let fields = self.data.as_ref().take_struct().unwrap();
        if self.is_tuple() {
            let fields: Vec<&str> = fields.iter().map(|_| "{:?}").collect();
            let fields = fields.join(", ");
            format!("{name}({fields})")
        } else {
            let fields: Vec<String> = fields
                .iter()
                .map(|field| {
                    let name = field.name.clone().unwrap_or_else(|| {
                        RenameRule::PascalCase
                            .apply_to_field(field.ident.as_ref().unwrap().to_string())
                    });

                    format!("{name}: {{:?}}")
                })
                .collect();
            let fields = fields.join(", ");
            format!("{name} - {fields}")
        }
    }

    fn generate_fmt(&self) -> TokenStream {
        let name = &self.ident;
        let (generics, generic_names, where_clause) = self.generics.split_for_impl();
        let fmt_str = self.generate_fmt_str();
        let fields = self.data.as_ref().take_struct().unwrap();
        let fmt_args = fields.iter().enumerate().map(|(i, field)| {
            if let Some(ident) = field.ident.as_ref() {
                quote![self.#ident]
            } else {
                let idx = Index::from(i);
                quote![self.#idx]
            }
        });
        let fmt_call = quote![write!(f, #fmt_str, #(#fmt_args),*)];
        quote! {
            impl #generics ::core::fmt::Display for #name #generic_names #where_clause {
                fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                    #fmt_call
                }
            }
        }
    }

    fn generate_new(&self) -> TokenStream {
        let name = &self.ident;
        let (generics, generic_names, where_clause) = self.generics.split_for_impl();
        let vis = &self.vis;
        let new_fn = match self.is_tuple() {
            true => self.generate_new_fn_tuple(),
            false => self.generate_new_fn_named(),
        };

        quote! {
            impl #generics #name #generic_names #where_clause {
                #[allow(clippy::too_many_arguments)]
                #vis #new_fn
            }
        }
    }

    fn generate_new_fn_named(&self) -> TokenStream {
        let fields = self.data.as_ref().take_struct().unwrap();
        let new_args = fields.iter().map(|it| {
            let name = it.ident.as_ref().unwrap();
            let ty = &it.ty;
            quote![#name: #ty]
        });
        let field_inits = fields.iter().map(|field| {
            let name = field.ident.as_ref().unwrap();
            let init = field_init(field, Member::Named(name.clone()));
            quote![#name: #init]
        });

        quote! {
            fn new(#(#new_args),*) -> Self {
                Self {
                    #(#field_inits),*
                }
            }
        }
    }

    fn generate_new_fn_tuple(&self) -> TokenStream {
        let fields = self.data.as_ref().take_struct().unwrap();
        let new_args = fields.iter().enumerate().map(|(i, field)| {
            let name = format_ident!("{i}_");
            let ty = &field.ty;
            quote![#name: #ty]
        });
        let field_inits = fields
            .iter()
            .enumerate()
            .map(|(i, field)| field_init(field, Member::Unnamed(Index::from(i))));

        quote! {
            fn new(#(#new_args),*) -> Self {
                Self (
                    #(#field_inits),*
                )
            }
        }
    }

    fn generate_key_impl(&self) -> TokenStream {
        let key = tune_type("AutotuneKey");
        let name = &self.ident;
        let (generics, generic_names, where_clause) = self.generics.split_for_impl();
        quote![impl #generics #key for #name #generic_names #where_clause {}]
    }
}

fn field_init(field: &AutotuneKeyField, member: Member) -> TokenStream {
    let anchor_fn = tune_type("anchor");
    field
        .anchor
        .as_ref()
        .map(|anchor| {
            let max = anchor.max();
            quote![#anchor_fn(#member, #max)]
        })
        .unwrap_or_else(|| member.to_token_stream())
}

pub fn generate_autotune_key(input: DeriveInput) -> syn::Result<TokenStream> {
    let key = AutotuneKey::from_derive_input(&input)?;
    let display = key.generate_fmt();
    let new = key.generate_new();
    let key_impl = key.generate_key_impl();
    Ok(quote! {
        #display
        #new
        #key_impl
    })
}

impl AutotuneOperations {
    fn generate_struct(&self) -> TokenStream {
        let name = &self.name;
        let generics = &self.generics;
        let where_clause = &generics.where_clause;
        let ty = self.ty.as_ref();
        let fields = &self.input_fields;
        quote! {
            #[derive(Debug)]
            pub struct #name #generics #where_clause {
                #ty
                #(#fields),*
            }
        }
    }

    fn generate_tunables_fn(&self) -> TokenStream {
        let operation = tune_type("AutotuneOperation");
        let output = &self.output;
        let key = &self.key;
        let generics = self.generics.split_for_impl();
        let generics = generics.1.as_turbofish();
        let fields = self.input_fields.iter().map(|field| {
            let name = field.ident.as_ref().unwrap();
            quote![let #name = &self.#name;]
        });
        let bench_inputs = self
            .input_fields
            .iter()
            .filter(|it| it.ident.as_ref().unwrap() != key)
            .map(|field| {
                let name = field.ident.as_ref().unwrap();
                quote![#name]
            })
            .collect::<Vec<_>>();
        let bench_ops = self.operations.iter().map(|op| {
            let op_struct = operation_name(op);
            quote![Box::new(#op_struct #generics::new(#(#bench_inputs.clone()),*))]
        });

        let body = &self.tunables_fn;
        quote! {
            #[allow(unused)]
            fn autotunables(&self) -> ::std::vec::Vec<Box<dyn #operation<#output>>> {
                #(#fields)*
                let (#(#bench_inputs),*) = #body;
                vec![
                    #(#bench_ops),*
                ]
            }
        }
    }

    fn generate_fastest_fn(&self) -> TokenStream {
        let operation = tune_type("AutotuneOperation");
        let output = &self.output;
        let (_, generics, _) = self.generics.split_for_impl();
        let generics = generics.as_turbofish();
        let fields: Vec<_> = self
            .input_fields
            .iter()
            .filter(|it| it.ident.as_ref().unwrap() != &self.key)
            .map(|field| {
                let name = field.ident.as_ref().unwrap();
                quote![self.#name.clone()]
            })
            .collect();
        let ops = self.operations.iter().enumerate().map(|(i, op)| {
            let op_struct = operation_name(op);
            quote![#i => Box::new(#op_struct #generics::new(#(#fields),*))]
        });

        quote! {
            fn fastest(
                self: Box<Self>,
                fastest_index: usize,
            ) -> Box<dyn #operation<#output>> {
                match fastest_index {
                    #(#ops,)*
                    _ => panic!("Fastest index is out of bound"),
                }
            }
        }
    }

    fn generate_autotune_impl(&self) -> TokenStream {
        let opset = tune_type("AutotuneOperationSet");

        let name = &self.name;
        let (generics, generic_names, where_clause) = self.generics.split_for_impl();
        let key = &self.key;
        let key_ty = &self.key_ty;
        let output = &self.output;
        let tunables = self.generate_tunables_fn();
        let fastest = self.generate_fastest_fn();
        let should_run = self.generate_should_run();

        quote! {
            impl #generics #opset<#key_ty, #output> for #name #generic_names #where_clause {
                fn key(&self) -> #key_ty {
                    self.#key.clone()
                }

                #tunables
                #fastest
                #should_run
            }
        }
    }

    fn generate_new(&self) -> TokenStream {
        if let Some(create_key) = self.create_key.as_ref() {
            let name = &self.name;
            let (generics, generic_names, where_clause) = self.generics.split_for_impl();
            let ty = self
                .ty
                .is_some()
                .then(|| quote![__ty: ::core::marker::PhantomData,]);
            let key = &self.key;
            let args = self
                .input_fields
                .iter()
                .filter(|it| it.ident.as_ref().unwrap() != key)
                .map(|field| {
                    let name = field.ident.as_ref().unwrap();
                    let ty = &field.ty;
                    quote![#name: #ty]
                });
            let field_names: Vec<_> = self
                .input_fields
                .iter()
                .filter(|it| it.ident.as_ref().unwrap() != key)
                .map(|field| {
                    let name = field.ident.as_ref().unwrap();
                    quote![#name]
                })
                .collect();

            quote! {
                impl #generics #name #generic_names #where_clause {
                    pub fn new(#(#args),*) -> Self {
                        Self {
                            #key: #create_key(#(&#field_names),*),
                            #ty
                            #(#field_names),*
                        }
                    }
                }
            }
        } else {
            TokenStream::new()
        }
    }

    fn generate_should_run(&self) -> TokenStream {
        if let Some(should_run) = self.should_run.as_ref() {
            let key_ty = &self.key_ty;

            quote! {
                fn should_run(&self, key: &#key_ty, index: usize) -> bool {
                    #should_run(self, key, index)
                }
            }
        } else {
            TokenStream::new()
        }
    }

    fn generate_operations(&self) -> TokenStream {
        let ops = self.operations.iter().map(|func_name| {
            let name = operation_name(func_name);
            let struct_ = self.generate_op_struct(&name);
            let impl_ = self.generate_op_impl(&name, func_name);
            let new = self.generate_op_new(&name);

            quote! {
                #struct_
                #impl_
                #new
            }
        });
        quote! {
            #(#ops)*
        }
    }

    fn generate_op_struct(&self, name: &Ident) -> TokenStream {
        let key = &self.key;
        let generics = &self.generics;
        let where_clause = &generics.where_clause;
        let ty = self.ty.as_ref();
        let fields = self
            .input_fields
            .iter()
            .filter(|it| it.ident.as_ref().unwrap() != key);
        quote! {
            #[derive(Debug)]
            pub struct #name #generics #where_clause {
                #ty
                #(#fields),*
            }
        }
    }

    fn generate_op_impl(&self, name: &Ident, func_name: &Path) -> TokenStream {
        let operation = tune_type("AutotuneOperation");

        let key = &self.key;
        let (generics, generic_names, where_clause) = self.generics.split_for_impl();
        let turbofish = generic_names.as_turbofish();
        let output = &self.output;
        let func_args = self
            .input_fields
            .iter()
            .filter(|it| it.ident.as_ref().unwrap() != key)
            .map(|field| {
                let name = field.ident.as_ref().unwrap();
                quote![self.#name]
            });
        let ty = self.ty.is_some().then(|| {
            quote! {
                __ty: ::core::marker::PhantomData,
            }
        });
        let clones = self
            .input_fields
            .iter()
            .filter(|it| it.ident.as_ref().unwrap() != key)
            .map(|field| {
                let name = field.ident.as_ref().unwrap();
                quote![#name: self.#name.clone()]
            });

        quote! {
            impl #generics #operation<#output> for #name #generic_names #where_clause {
                fn execute(self: Box<Self>) -> #output {
                    #func_name #turbofish(#(#func_args),*)
                }

                fn clone(&self) -> Box<dyn #operation<#output>> {
                    Box::new(Self {
                        #ty
                        #(#clones),*
                    })
                }
            }
        }
    }

    fn generate_op_new(&self, name: &Ident) -> TokenStream {
        let key = &self.key;
        let (generics, generic_names, where_clause) = self.generics.split_for_impl();
        let ty = self
            .ty
            .is_some()
            .then(|| quote![__ty: ::core::marker::PhantomData,]);
        let args = self
            .input_fields
            .iter()
            .filter(|it| it.ident.as_ref().unwrap() != key)
            .map(|field| {
                let name = field.ident.as_ref().unwrap();
                let ty = &field.ty;
                quote![#name: #ty]
            });
        let field_names = self
            .input_fields
            .iter()
            .filter(|it| it.ident.as_ref().unwrap() != key)
            .map(|field| {
                let name = field.ident.as_ref().unwrap();
                quote![#name]
            });

        quote! {
            impl #generics #name #generic_names #where_clause {
                pub fn new(#(#args),*) -> Self {
                    Self {
                        #ty
                        #(#field_names),*
                    }
                }
            }
        }
    }
}

pub fn generate_autotune_set(
    item: ItemFn,
    args: AutotuneOperationsArgs,
) -> syn::Result<TokenStream> {
    let opset = AutotuneOperations::from_item_fn(item, args)?;
    let struct_ = opset.generate_struct();
    let impl_ = opset.generate_autotune_impl();
    let new = opset.generate_new();
    let ops = opset.generate_operations();

    Ok(quote! {
        #struct_
        #impl_
        #new
        #ops
    })
}
