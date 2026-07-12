use core::iter;

use darling::{FromDeriveInput};
use inflections::case::{to_constant_case, to_snake_case};
use proc_macro2::TokenStream;
use quote::{ToTokens, format_ident, quote};
use syn::{DeriveInput, Ident, Type};

use crate::{
    CubeOp, CubeOpArgs, PathList, parse::cube_op::{ArgKind, CubeOpArg, ResultTy, Verifier},
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
        let name = args.name.value();
        let format = if self.custom_format(&args) {
            quote![]
        } else {
            let format = self.qualified_format_string(&args).map(|fmt| quote![,format = #fmt]);
            format.unwrap_or_else(|| quote![,format])
        };
        
        let verifier = &args.verifier;
        let attributes = self
            .attributes()
            .map(|CubeOpArg { ident, ty, flags, .. }| {
                let ident = args.qualified_name(ident);
                if flags.untyped.is_present() {
                    quote![#ident]
                } else {
                    quote![#ident: #ty]
                }
            })
            .collect::<Vec<_>>();
        let attributes = if attributes.is_empty() {
            quote![]
        } else {
            quote![attributes = (#(#attributes),*),]
        };

        let interfaces = self.auto_interfaces();
        let constructor = self.generate_constructor(&args);
        let accessors = self.generate_value_accessors(&args);
        let setters = self.generate_attribute_setters(&args);
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
            #[::pliron::derive::pliron_op(name = #name #format, #attributes #verifier)]
            #[::pliron::derive::derive_op_interface_impl(#(#interfaces),*)]
            #vis struct #ident #generics;

            impl #impl_generics #ident #type_generics #where_clause {
                #constructor
                #(#accessors)*
                #(#setters)*
            }

            #(#memory_interfaces)*
        })
    }

    fn generate_constructor(&self, op_args: &CubeOpArgs) -> TokenStream {
        let result_ty = self.generate_result_ty();
        let args = self
            .data
            .iter()
            .map(|arg| {
                let CubeOpArg { ident, ty, kind, .. } = arg;
                match (kind, arg.flags.optional.is_present()) {
                    (_, true) => quote![#ident: Option<#ty>],
                    (ArgKind::Value, false) => quote![#ident: #ty],
                    (ArgKind::Attribute, false) => quote![#ident: impl Into<#ty>]
                }
            });
        let values = self.values().map(|arg| &arg.ident);
        let attr_into = self.required_attributes().map(|it| {
            let name = &it.ident;
            quote![let #name = #name.into();]
        });
        let attributes = self.required_attributes().map(|it| {
            let name = &it.ident;
            let setter = format_ident!("set_attr_{}", op_args.qualified_name(name));
            quote![op.#setter(ctx, #name);]
        });
        let opt_attrs = self.optional_attributes().map(|it| {
            let name = &it.ident;
            let setter = format_ident!("set_attr_{}", op_args.qualified_name(name));
            quote![if let Some(attr) = #name {
                op.#setter(ctx, attr);
            }]
        });

        let args: Vec<_> = match self.result_ty {
            ResultTy::Argument => iter::once(quote![result_ty: ::pliron::r#type::TypeHandle])
                .chain(args)
                .collect(),
            _ => args.collect(),
        };

        quote! {
            #[allow(clippy::too_many_arguments)]
            pub fn new(ctx: &mut ::pliron::context::Context, #(#args),*) -> Self {
                use ::pliron::{r#type::Typed, op::Op};
                #(#attr_into)*
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
                #(#opt_attrs)*
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

    fn generate_value_accessors(&self, args: &CubeOpArgs) -> impl Iterator<Item = TokenStream> {
        let values = self.values().enumerate().map(|(idx, arg)| {
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
        });
        let attrs_req = self.required_attributes().map(|arg| {
            let CubeOpArg { vis, ident, ty, .. } = arg;
            let inner_accessor = format_ident!("get_attr_{}", args.qualified_name(ident));
            quote! {
                #vis fn #ident<'a>(&self, ctx: &'a ::pliron::context::Context) -> core::cell::Ref<'a, #ty> {
                    self.#inner_accessor(ctx).unwrap()
                }
            }
        });
        let attrs_opt = self.optional_attributes().map(|arg| {
            let CubeOpArg { vis, ident, ty, .. } = arg;
            let inner_accessor = format_ident!("get_attr_{}", args.qualified_name(ident));
            if let Type::Path(path) = ty && path.path.segments.last().unwrap().ident == "UnitAttr" {
                quote! {
                    #vis fn #ident<'a>(&self, ctx: &'a ::pliron::context::Context) -> bool {
                        self.#inner_accessor(ctx).is_some()
                    }
                }
            } else {
                quote! {
                    #vis fn #ident<'a>(&self, ctx: &'a ::pliron::context::Context) -> Option<core::cell::Ref<'a, #ty>> {
                        self.#inner_accessor(ctx)
                    }
                }
            }
        });
        values.chain(attrs_req).chain(attrs_opt)
    }

    fn generate_attribute_setters(&self, args: &CubeOpArgs) -> impl Iterator<Item = TokenStream> {
        let attrs_set = self.attributes().map(|arg| {
            let CubeOpArg { vis, ident, ty, .. } = arg;
            let setter = format_ident!("set_{ident}");
            let inner_accessor = format_ident!("set_attr_{}", args.qualified_name(ident));
            if let Type::Path(path) = ty && path.path.segments.last().unwrap().ident == "UnitAttr" {
                quote! {
                    #vis fn #setter(&self, ctx: &::pliron::context::Context) {
                        self.#inner_accessor(ctx, #ty::new())
                    }
                }
            } else {
                quote! {
                    #vis fn #setter(&self, ctx: &::pliron::context::Context, value: impl Into<#ty>) {
                        self.#inner_accessor(ctx, value.into())
                    }
                }
            }
        });
        let attrs_remove = self.optional_attributes().map(|arg| {
            let CubeOpArg { vis, ident, .. } = arg;
            let remove = format_ident!("remove_{ident}");
            let attr_key = self.qualified_attribute_key(ident, args);
            quote! {
                #vis fn #remove(&self, ctx: &::pliron::context::Context) {
                    self.get_operation().deref_mut(ctx).attributes.0.remove(&*#attr_key);
                }
            }
        });
        attrs_set.chain(attrs_remove)
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

        let ptr_reads = args.iter().filter(|arg| arg.flags.ptr_read.is_present()).map(|arg| {
            let ident = &arg.ident;
            quote![crate::interfaces::MemoryEffect::Read(self.#ident(ctx))]
        });
        let ptr_writes = args.iter().filter(|arg| arg.flags.ptr_write.is_present()).map(|arg| {
            let ident = &arg.ident;
            quote![crate::interfaces::MemoryEffect::Write(self.#ident(ctx))]
        });
        let memory_effects = ptr_reads.chain(ptr_writes).collect::<Vec<_>>();

        if !memory_effects.is_empty() {
            interfaces.push(quote! {
                #[::pliron::derive::op_interface_impl]
                impl #impl_generics crate::interfaces::MemoryEffects for #ty_name #type_generics #where_clause {
                    fn memory_effects(&self, ctx: &Context) -> ::alloc::vec::Vec<crate::interfaces::MemoryEffect> {
                        ::alloc::vec![#(#memory_effects),*]
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

    fn optional_attributes(&self) -> impl Iterator<Item = &CubeOpArg> {
        self.attributes()
            .filter(|it| it.flags.optional.is_present())
    }

    fn qualified_format_string(&self, args: &CubeOpArgs) -> Option<String> {
        let mut format = args.format.as_ref()?.value();
        for attr in self.attributes() {
            let attr_name_ref = format!("attr(${}", attr.ident);
            let qualified = format!("attr(${}", args.qualified_name(&attr.ident));
            format = format.replace(&attr_name_ref, &qualified);
        }
        Some(format)
    }

    fn custom_format(&self, args: &CubeOpArgs) -> bool {
        args.format.as_ref().is_some_and(|it| it.value() == "custom")
    }

    fn qualified_attribute_key(&self, ident: &Ident, args: &CubeOpArgs) -> TokenStream {
        let op_name = self.ident.clone();
        let module_name = format_ident!("{}_attr_names", to_snake_case(&op_name.to_string()));
        let attr_name_uppercase = to_constant_case(&args.qualified_name(ident).to_string());
        let attr_name_const = format_ident!("ATTR_KEY_{}", attr_name_uppercase);

        quote![#module_name::#attr_name_const]
    }
}

impl CubeOpArgs {
    fn qualified_name(&self, ident: &Ident) -> Ident {
        let name = self.name.value();
        let qualification = name.replace(".", "_");
        format_ident!("{qualification}_{ident}")
    }
}

fn op_interf(name: TokenStream) -> TokenStream {
    quote![::pliron::builtin::op_interfaces::#name]
}

pub fn generate_cube_op(input: DeriveInput, args: CubeOpArgs) -> syn::Result<TokenStream> {
    let op = CubeOp::from_derive_input(&input)?;
    op.generate_op_impl(args)
}

pub fn generate_op_traits(input: DeriveInput, traits: PathList) -> syn::Result<TokenStream> {
    let traits = &traits.paths;
    let struct_name = &input.ident;
    Ok(quote::quote! {
        #input
        #(#traits !(#struct_name);)*
    })
}