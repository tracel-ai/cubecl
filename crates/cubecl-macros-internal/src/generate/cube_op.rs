use core::iter;

use darling::FromDeriveInput;
use proc_macro2::TokenStream;
use quote::{ToTokens, format_ident, quote};
use syn::DeriveInput;

use crate::{
    CubeOp, CubeOpArgs,
    parse::cube_op::{ArgKind, CubeOpArg, ResultTy, Verifier},
};

impl ToTokens for Verifier {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            Verifier::Succ => {
                tokens.extend(quote![verifier = "succ"]);
            }
            Verifier::Custom => {}
        }
    }
}

impl CubeOp {
    pub fn generate_op_impl(&self, args: CubeOpArgs) -> syn::Result<TokenStream> {
        let name = &args.name;
        let format = args.format.as_ref().map(|fmt| quote![format = #fmt]);
        let format = format.unwrap_or_else(|| quote![format]);
        let verifier = &args.verifier;
        let attributes = self
            .attributes()
            .map(|CubeOpArg { ident, ty, .. }| quote![#ident: #ty])
            .collect::<Vec<_>>();
        let attributes = if attributes.is_empty() {
            quote![]
        } else {
            quote![attributes = (#(#attributes),*),]
        };

        let interfaces = self.auto_interfaces();
        let constructor = self.generate_constructor();
        let accessors = self.generate_value_accessors();
        let memory_interfaces = self.generate_memory_interfaces();

        let Self {
            vis,
            ident,
            generics,
            attrs,
            ..
        } = self;

        let (impl_generics, type_generics, where_clause) = generics.split_for_impl();

        Ok(quote! {
            #(#attrs)*
            #[::pliron::derive::pliron_op(name = #name, #format, #attributes #verifier)]
            #[::pliron::derive::derive_op_interface_impl(#(#interfaces),*)]
            #vis struct #ident #generics;

            impl #impl_generics #ident #type_generics #where_clause {
                #constructor
                #(#accessors)*
            }

            #(#memory_interfaces)*
        })
    }

    fn generate_constructor(&self) -> TokenStream {
        let result_ty = self.generate_result_ty();
        let args = self
            .data
            .iter()
            .filter(|it| !it.flags.optional.is_present())
            .map(|arg| {
                let CubeOpArg { ident, ty, .. } = arg;
                quote![#ident: #ty]
            });
        let values = self.values().map(|arg| &arg.ident);
        let attributes = self.required_attributes().map(|it| {
            let name = &it.ident;
            let setter = format_ident!("set_attr_{name}");
            quote![op.#setter(ctx, #name);]
        });

        let args: Vec<_> = match self.result_ty {
            ResultTy::Argument => iter::once(quote![result_ty: Ptr<TypeObj>])
                .chain(args)
                .collect(),
            _ => args.collect(),
        };

        quote! {
            #[allow(clippy::too_many_arguments)]
            pub fn new(ctx: &mut ::pliron::context::Context, #(#args),*) -> Self {
                use ::pliron::{r#type::Typed, op::Op};
                let result_ty = #result_ty;
                let values = vec![#(#values),*];
                let op = Self {
                    op: Operation::new(
                        ctx,
                        Self::get_concrete_op_info(),
                        result_ty,
                        values,
                        vec![],
                        0
                    )
                };
                #(#attributes)*
                op
            }
        }
    }

    fn generate_result_ty(&self) -> TokenStream {
        let ty = match &self.result_ty {
            ResultTy::None => quote![],
            ResultTy::SameAs(input) => quote![#input.get_type(ctx)],
            ResultTy::Fixed(expr) => quote![#expr],
            ResultTy::FromInputs(expr) => {
                let args = self
                    .data
                    .iter()
                    .filter(|arg| !arg.flags.optional.is_present())
                    .map(|arg| &arg.ident);
                quote![(#expr)(ctx, #(&#args),*)]
            }
            ResultTy::Argument => {
                quote![result_ty]
            }
        };
        quote!(vec![#ty])
    }

    fn generate_value_accessors(&self) -> impl Iterator<Item = TokenStream> {
        self.values().enumerate().map(|(idx, arg)| {
            let CubeOpArg { vis, ident, ty, .. } = arg;
            let use_ident = format_ident!("{ident}_as_use");
            quote! {
                #vis fn #ident(&self, ctx: &::pliron::context::Context) -> #ty {
                    self.get_operation().deref(ctx).get_operand(#idx)
                }
                
                #vis fn #use_ident(&self, ctx: &::pliron::context::Context) -> ::pliron::value::Use<#ty> {
                    self.get_operation().deref(ctx).get_operand_as_use(#idx)
                }
            }
        })
    }

    fn auto_interfaces(&self) -> Vec<TokenStream> {
        let mut interfaces = Vec::new();
        let num_opds = self.values().count();

        interfaces.push(op_interf(quote![AtMostNOpdsInterface<#num_opds>]));
        interfaces.push(op_interf(quote![NOpdsInterface<#num_opds>]));
        if num_opds == 1 {
            interfaces.push(op_interf(quote![OneOpdInterface]));
        }
        if num_opds > 0 {
            interfaces.extend((1..=num_opds).map(|i| op_interf(quote![AtLeastNOpdsInterface<#i>])));
        }

        if self.result_ty.has_result() {
            interfaces.push(op_interf(quote![AtLeastNResultsInterface<1>]));
            interfaces.push(op_interf(quote![AtMostNResultsInterface<1>]));
            interfaces.push(op_interf(quote![NResultsInterface<1>]));
            interfaces.push(op_interf(quote![OneResultInterface]));
            interfaces.push(op_interf(quote![SameResultsType]));
        } else {
            interfaces.push(op_interf(quote![AtMostNResultsInterface<0>]));
            interfaces.push(op_interf(quote![NResultsInterface<0>]));
        }

        interfaces
    }

    fn generate_memory_interfaces(&self) -> Vec<TokenStream> {
        let mut interfaces = vec![];

        let ty_name = &self.ident;
        let (impl_generics, type_generics, where_clause) = self.generics.split_for_impl();
        let args = &self.data;

        if args.iter().any(|arg| arg.flags.ptr_read.is_present()) {
            let values = args
                .iter()
                .filter(|arg| arg.flags.ptr_read.is_present())
                .map(|CubeOpArg { ident, .. }| quote![self.#ident(ctx)]);
            interfaces.push(quote! {
                #[::pliron::derive::op_interface_impl]
                impl #impl_generics crate::interfaces::ReadsMemory for #ty_name #type_generics #where_clause {
                    fn reads_through_values(&self, ctx: &Context) -> ::alloc::vec::Vec<Value> {
                        ::alloc::vec![#(#values),*]
                    }
                }
            });
        }

        if args.iter().any(|arg| arg.flags.ptr_write.is_present()) {
            let values = args
                .iter()
                .filter(|arg| arg.flags.ptr_write.is_present())
                .map(|CubeOpArg { ident, .. }| quote![self.#ident(ctx)]);
            interfaces.push(quote! {
                #[::pliron::derive::op_interface_impl]
                impl #impl_generics crate::interfaces::WritesMemory for #ty_name #type_generics #where_clause {
                    fn writes_through_values(&self, ctx: &Context) -> ::alloc::vec::Vec<Value> {
                        ::alloc::vec![#(#values),*]
                    }
                }
            });
        }

        interfaces
    }

    fn values(&self) -> impl Iterator<Item = &CubeOpArg> {
        self.data
            .iter()
            .filter(|it| matches!(it.kind, ArgKind::Value))
    }

    fn attributes(&self) -> impl Iterator<Item = &CubeOpArg> {
        self.data
            .iter()
            .filter(|it| matches!(it.kind, ArgKind::Attribute))
    }

    fn required_attributes(&self) -> impl Iterator<Item = &CubeOpArg> {
        self.attributes()
            .filter(|it| !it.flags.optional.is_present())
    }
}

fn op_interf(name: TokenStream) -> TokenStream {
    quote![::pliron::builtin::op_interfaces::#name]
}

pub fn generate_cube_op(input: DeriveInput, args: CubeOpArgs) -> syn::Result<TokenStream> {
    let op = CubeOp::from_derive_input(&input)?;
    op.generate_op_impl(args)
}
