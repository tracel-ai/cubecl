use ident_case::RenameRule;
use proc_macro2::TokenStream;
use quote::{format_ident, quote, quote_spanned, ToTokens};
use syn::{parse_quote, spanned::Spanned as _, Ident};

use crate::{
    parse::kernel::{KernelParam, Launch},
    paths::{core_path, core_type, ir_type, prelude_type},
};

impl ToTokens for Launch {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let vis = &self.vis;

        let name = &self.func.sig.name;
        let launch = self.launch();
        let launch_unchecked = self.launch_unchecked();
        let dummy = self.create_dummy_kernel();
        let kernel = self.kernel_definition();
        let checks = self.check_args();
        let mut func = self.func.clone();
        func.sig.name = format_ident!("expand");

        let out = quote! {
            #vis mod #name {
                use super::*;

                #[allow(unused, clippy::all)]
                pub #func

                #kernel
                #launch
                #launch_unchecked
                #dummy
                #checks
            }
        };

        if self.args.debug.is_present() {
            let file = syn::parse_file(&out.to_string()).unwrap();
            let tokens = prettyplease::unparse(&file);
            panic!("{tokens}");
        }
        tokens.extend(out);
    }
}

impl Launch {
    fn launch(&self) -> TokenStream {
        if self.args.launch.is_present() {
            let compute_client = prelude_type("ComputeClient");
            let cube_count = prelude_type("CubeCount");
            let cube_dim = prelude_type("CubeDim");

            let kernel_doc = format!(
                "Launch the kernel [{}()] on the given runtime",
                self.func.sig.name
            );
            let generics = &self.launch_generics;
            let args = self.launch_args();
            let body = self.launch_body();

            quote! {
                #[allow(clippy::too_many_arguments)]
                #[doc = #kernel_doc]
                pub fn launch #generics(
                    __client: &#compute_client<__R::Server, __R::Channel>,
                    __cube_count: #cube_count<__R::Server>,
                    __cube_dim: #cube_dim,
                    #(#args),*
                ) -> () {
                    #body
                    launcher.launch(__cube_count, kernel, __client);
                }
            }
        } else {
            TokenStream::new()
        }
    }

    fn launch_unchecked(&self) -> TokenStream {
        if self.args.launch_unchecked.is_present() {
            let compute_client = prelude_type("ComputeClient");
            let cube_count = prelude_type("CubeCount");
            let cube_dim = prelude_type("CubeDim");

            let kernel_doc = format!(
                "Launch the kernel [{}()] on the given runtime",
                self.func.sig.name
            );
            let generics = &self.launch_generics;
            let args = self.launch_args();
            let body = self.launch_body();

            quote! {
                #[allow(clippy::too_many_arguments)]
                #[doc = #kernel_doc]
                pub unsafe fn launch_unchecked #generics(
                    __client: &#compute_client<__R::Server, __R::Channel>,
                    __cube_count: #cube_count<__R::Server>,
                    __cube_dim: #cube_dim,
                    #(#args),*
                ) -> () {
                    #body
                    launcher.launch_unchecked(__cube_count, kernel, __client);
                }
            }
        } else {
            TokenStream::new()
        }
    }

    fn launch_body(&self) -> TokenStream {
        let kernel_launcher = prelude_type("KernelLauncher");

        let registers = self.runtime_params().map(|arg| {
            let name = &arg.name;
            quote![#name.register(&mut launcher);]
        });

        let settings = self.configure_settings();
        let kernel_name = self.kernel_name();
        let core_path = core_path();
        let comptime_args = self.comptime_params().map(|it| &it.name);

        quote! {
            use #core_path::frontend::ArgSettings as _;

            #settings
            let kernel = #kernel_name::new(__settings, #(#comptime_args),*);
            let mut launcher = #kernel_launcher::<__R>::default();
            #(#registers)*
        }
    }

    fn configure_settings(&self) -> TokenStream {
        let kernel_settings = prelude_type("KernelSettings");
        let arg_settings = prelude_type("ArgSettings");

        let input_configs = self.runtime_inputs().enumerate().map(|(i, arg)| {
            let name = &arg.name;
            quote![__settings = #arg_settings::<__R>::configure_input(&#name, #i, __settings);]
        });
        let output_configs = self.runtime_outputs().enumerate().map(|(i, arg)| {
            let name = &arg.name;
            quote![__settings = #arg_settings::<__R>::configure_output(&#name, #i, __settings);]
        });

        quote! {
            let mut __settings = #kernel_settings::default().cube_dim(__cube_dim);
            #(#input_configs)*
            #(#output_configs)*
        }
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
                &input.ty,
            )
        });
        let outputs = self.runtime_outputs().enumerate().map(|(i, output)| {
            expand_fn(
                i,
                format_ident!("expand_output"),
                format_ident!("vectorization_output"),
                &output.ty,
            )
        });
        let map = quote![::std::collections::BTreeMap<usize, std::sync::Arc<dyn core::any::Any>> = std::collections::BTreeMap::new()];
        let inputs_len = self.runtime_inputs().count();
        let outputs_len = self.runtime_outputs().count();
        let register_input = register_fn("register_input", inputs);
        let register_output = register_fn("register_output", outputs);

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

            for i in 0..#inputs_len {
                inputs.insert(i, register_input(&mut builder, &self.settings, i));
            }
            for mapping in &self.settings.mappings {
                let input = inputs.get(&mappings.pos_input).unwrap();
                outputs.insert(mapping.pos_output, input.clone());
            }
            for i in 0..#outputs_len {
                if !outputs.contains_key(&i) {
                    outputs.insert(i, register_output(&mut builder, &self.settings, i));
                }
            }
            #(#in_params)*
            #(#out_params)*
        }
    }

    fn create_dummy_kernel(&self) -> TokenStream {
        if self.args.create_dummy_kernel.is_present() {
            let cube_count = prelude_type("CubeCount");
            let cube_dim = prelude_type("CubeDim");

            let kernel_doc = format!(
                "Launch the kernel [{}()] on the given runtime",
                self.func.sig.name
            );
            let (generics, generic_names, _) = self.kernel_generics.split_for_impl();

            let settings = self.configure_settings();
            let kernel_name = self.kernel_name();
            let core_path = core_path();
            let comptime_args = self.comptime_params();
            let comptime_names = self.comptime_params().map(|it| &it.name);

            quote! {
                #[allow(clippy::too_many_arguments)]
                #[doc = #kernel_doc]
                pub fn create_dummy_kernel #generics(
                    __cube_count: #cube_count<__R::Server>,
                    __cube_dim: #cube_dim,
                    #(#comptime_args),*
                ) -> #kernel_name #generic_names {
                    use #core_path::frontend::ArgSettings as _;

                    #settings
                    #kernel_name::new(__settings, #(#comptime_names),*);
                }
            }
        } else {
            TokenStream::new()
        }
    }

    pub fn runtime_inputs(&self) -> impl Iterator<Item = &KernelParam> {
        self.runtime_params().filter(|it| !it.is_mut)
    }

    pub fn runtime_outputs(&self) -> impl Iterator<Item = &KernelParam> {
        self.runtime_params().filter(|it| it.is_mut)
    }

    pub fn runtime_params(&self) -> impl Iterator<Item = &KernelParam> {
        self.func.sig.parameters.iter().filter(|it| !it.is_const)
    }

    fn launch_args(&self) -> Vec<KernelParam> {
        let mut args = self.func.sig.parameters.clone();
        let runtime_arg = core_type("RuntimeArg");
        for arg in args.iter_mut().filter(|it| !it.is_const) {
            let ty = arg.ty_owned();
            arg.normalized_ty = parse_quote![#runtime_arg<'kernel, #ty, __R>];
        }
        args
    }

    pub fn kernel_name(&self) -> Ident {
        let kernel_name = RenameRule::PascalCase.apply_to_field(self.func.sig.name.to_string());
        format_ident!("{kernel_name}")
    }

    pub fn comptime_params(&self) -> impl Iterator<Item = &KernelParam> {
        self.func
            .sig
            .parameters
            .iter()
            .filter(|param| param.is_const)
    }

    fn check_args(&self) -> TokenStream {
        if self.args.is_launch() {
            let generics = &self.func.sig.generics;

            let input_checks = self
                .func
                .sig
                .parameters
                .iter()
                // Const can be anything as long as the accessed fields are cube types, since the access
                // gets resolved at expansion time and collapsed into a literal in the kernel
                .filter(|arg| !arg.is_const)
                .map(|arg| {
                    let span = arg.ty.span();
                    let check = ir_type("assert_valid_type");
                    let ty = arg.ty_owned();
                    quote_spanned! {span=>
                        #check::<#ty>();
                    }
                })
                .collect::<Vec<_>>();

            quote! {
                fn __check_inputs #generics() {
                    #(#input_checks)*
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
        fn #name(
            builder: &mut #kernel_builder,
            settings: &#kernel_settings,
            position: usize,
        ) -> ::std::sync::Arc<dyn ::core::any::Any> {
            match position {
                #(#values,)*
                _ => {
                    panic!("Input {position} is invalid");
                }
            }
        }
    }
}

fn runtime_param(io_map: &str) -> impl FnMut((usize, &KernelParam)) -> TokenStream {
    let cube_type = prelude_type("CubeType");
    let io_map = format_ident!("{io_map}");
    move |(i, input)| {
        let name: &Ident = &input.name;
        let ty = &input.ty;
        quote! {
            let #name: &<#ty as #cube_type>::ExpandType = #io_map.get(&#i).unwrap().downcast_ref()
                .expect("Output type should be correct. It could be caused by an invalid kernel input/output alias.");
        }
    }
}
