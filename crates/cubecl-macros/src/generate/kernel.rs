use darling::usage::{CollectLifetimes as _, CollectTypeParams as _, GenericsExt as _, Purpose};
use proc_macro2::TokenStream;
use quote::{format_ident, quote, ToTokens};
use std::iter;
use syn::Ident;

use crate::{
    parse::kernel::{KernelFn, KernelParam, KernelSignature, Launch},
    paths::{core_type, prelude_path, prelude_type},
};

impl KernelFn {
    pub fn to_tokens_mut(&mut self) -> TokenStream {
        let prelude_path = prelude_path();
        let sig = &self.sig;
        let block = self.block.to_tokens(&mut self.context);

        let out = quote! {
            #sig {
                use #prelude_path::IntoRuntime as _;

                #block
            }
        };

        out
    }
}

impl ToTokens for KernelSignature {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let cube_context = prelude_type("CubeContext");
        let cube_type = prelude_type("CubeType");

        let name = &self.name;
        let generics = &self.generics;
        let return_type = &self.returns;
        let args = &self.parameters;

        let out = quote! {
            fn #name #generics(
                context: &mut #cube_context,
                #(#args),*
            ) -> <#return_type as #cube_type>::ExpandType
        };
        tokens.extend(out);
    }
}

impl ToTokens for KernelParam {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let name = &self.name;
        let ty = &self.normalized_ty;
        tokens.extend(quote![#name: #ty]);
    }
}

impl Launch {
    fn kernel_phantom_data(&self) -> Option<TokenStream> {
        let generics = self.kernel_generics.clone();
        let declared_lifetimes = generics.declared_lifetimes();
        let declared_type_params = generics.declared_type_params();

        let used_lifetimes = self
            .comptime_params()
            .map(|param| &param.ty)
            .collect_lifetimes_cloned(&Purpose::Declare.into(), &declared_lifetimes);
        let used_type_params = self
            .comptime_params()
            .map(|param| &param.ty)
            .collect_type_params_cloned(&Purpose::Declare.into(), &declared_type_params);
        let lifetimes: Vec<_> = declared_lifetimes.difference(&used_lifetimes).collect();
        let type_params: Vec<_> = declared_type_params.difference(&used_type_params).collect();

        (!lifetimes.is_empty() || !type_params.is_empty())
            .then(|| quote![__ty: ::core::marker::PhantomData<(#(#lifetimes,)* #(#type_params),*)>])
    }

    pub fn io_mappings(&self) -> TokenStream {
        let launch_arg_expand = prelude_type("LaunchArgExpand");
        let expand_fn = |i, expand_name, vec_name, ty| {
            quote! {
                #i => ::std::sync::Arc::new(<#ty as #launch_arg_expand>::#expand_name(builder, settings.#vec_name(#i)))
            }
        };
        let inputs = self.runtime_inputs().enumerate().map(|(i, input)| {
            expand_fn(
                i,
                format_ident!("expand"),
                format_ident!("vectorization_input"),
                input.ty_owned(),
            )
        });
        let outputs = self.runtime_outputs().enumerate().map(|(i, output)| {
            expand_fn(
                i,
                format_ident!("expand_output"),
                format_ident!("vectorization_output"),
                output.ty_owned(),
            )
        });
        let map = quote![::std::collections::BTreeMap<usize, std::sync::Arc<dyn core::any::Any>> = std::collections::BTreeMap::new()];
        let inputs_len = self.runtime_inputs().count();
        let outputs_len = self.runtime_outputs().count();
        let register_input = register_fn("register_input", inputs);
        let register_output = register_fn("register_output", outputs);

        let insert_inputs = (inputs_len > 0).then(|| {
            quote! {
                for i in 0..#inputs_len {
                    inputs.insert(i, register_input(&mut builder, &self.settings, i));
                }
            }
        });
        let insert_outputs = (outputs_len > 0).then(|| {
            quote! {
                for i in 0..#outputs_len {
                    if !outputs.contains_key(&i) {
                        outputs.insert(i, register_output(&mut builder, &self.settings, i));
                    }
                }
            }
        });

        let in_params = self
            .runtime_inputs()
            .enumerate()
            .map(runtime_param("inputs"));
        let out_params = self
            .runtime_outputs()
            .enumerate()
            .map(runtime_param("outputs"));

        quote! {
            let mut inputs: #map;
            let mut outputs: #map;

            #register_input
            #register_output

            #insert_inputs
            for mapping in &self.settings.mappings {
                let input = inputs.get(&mapping.pos_input).unwrap();
                outputs.insert(mapping.pos_output, input.clone());
            }
            #insert_outputs
            #(#in_params)*
            #(#out_params)*
        }
    }

    fn define_body(&self) -> TokenStream {
        let kernel_builder = prelude_type("KernelBuilder");
        let runtime = prelude_type("Runtime");
        let compiler = core_type("Compiler");
        let io_map = self.io_mappings();
        let runtime_args = self.runtime_params().map(|it| &it.name);
        let comptime_args = self.comptime_params().map(|it| &it.name);
        let (_, generics, _) = self.func.sig.generics.split_for_impl();
        let generics = generics.as_turbofish();

        quote! {
            let mut builder = #kernel_builder::with_local_allocator(<<__R as #runtime>::Compiler as #compiler>::local_allocator());
            #io_map
            expand #generics(&mut builder.context, #(#runtime_args.clone(),)* #(self.#comptime_args.clone()),*);
            builder.build(self.settings.clone())
        }
    }

    pub fn kernel_definition(&self) -> TokenStream {
        if self.args.is_launch() {
            let kernel = core_type("Kernel");
            let kernel_settings = prelude_type("KernelSettings");
            let kernel_definition: syn::Path = prelude_type("KernelDefinition");
            let kernel_id = core_type("KernelId");

            let kernel_name = self.kernel_name();
            let define = self.define_body();
            let kernel_doc = format!("{} Kernel", self.func.sig.name);

            let (generics, generic_names, where_clause) = self.kernel_generics.split_for_impl();
            let const_params: Vec<_> = self.comptime_params().collect();
            let param_names = self
                .comptime_params()
                .map(|param| param.name.clone())
                .collect::<Vec<_>>();
            let phantom_data = self.kernel_phantom_data();
            let info = iter::once(format_ident!("settings")).chain(param_names.clone());
            let phantom_data_init = phantom_data
                .as_ref()
                .map(|_| quote![__ty: ::core::marker::PhantomData]);

            quote! {
                #[doc = #kernel_doc]
                pub struct #kernel_name #generics #where_clause {
                    settings: #kernel_settings,
                    #(#const_params,)*
                    #phantom_data
                }

                impl #generics #kernel_name #generic_names #where_clause {
                    pub fn new(settings: #kernel_settings, #(#const_params),*) -> Self {
                        Self {
                            settings,
                            #(#param_names,)*
                            #phantom_data_init
                        }
                    }
                }

                impl #generics #kernel for #kernel_name #generic_names #where_clause {
                    fn define(&self) -> #kernel_definition {
                        #define
                    }

                    fn id(&self) -> #kernel_id {
                        #kernel_id::new::<Self>().info((#(self.#info.clone()),*))
                    }
                }
            }
        } else {
            TokenStream::new()
        }
    }
}

fn register_fn(name: &str, values: impl Iterator<Item = TokenStream>) -> TokenStream {
    let kernel_settings = prelude_type("KernelSettings");
    let kernel_builder = prelude_type("KernelBuilder");

    let name = format_ident!("{name}");
    quote! {
        #[allow(unused)]
        let #name = |
            builder: &mut #kernel_builder,
            settings: &#kernel_settings,
            position: usize,
        | -> ::std::sync::Arc<dyn ::core::any::Any> {
            match position {
                #(#values,)*
                _ => {
                    panic!("Input {position} is invalid");
                }
            }
        };
    }
}

fn runtime_param(io_map: &str) -> impl FnMut((usize, &KernelParam)) -> TokenStream {
    let cube_type = prelude_type("CubeType");
    let io_map = format_ident!("{io_map}");
    move |(i, input)| {
        let name: &Ident = &input.name;
        let ty = input.ty_owned();
        quote! {
            let #name: &<#ty as #cube_type>::ExpandType = #io_map.get(&#i).unwrap().downcast_ref()
                .expect("Output type should be correct. It could be caused by an invalid kernel input/output alias.");
        }
    }
}
