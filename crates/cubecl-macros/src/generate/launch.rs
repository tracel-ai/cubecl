use ident_case::RenameRule;
use proc_macro2::TokenStream;
use quote::{ToTokens, format_ident, quote};
use syn::{Ident, parse_quote};

use crate::{
    parse::kernel::{KernelParam, Launch},
    paths::{core_path, core_type, prelude_type},
};

impl ToTokens for Launch {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let vis = &self.vis;

        let name = &self.func.sig.name;
        let launch = self.launch();
        let launch_unchecked = self.launch_unchecked();
        let dummy = self.create_dummy_kernel();
        let kernel = self.kernel_definition();
        let mut func = self.func.clone();
        func.sig.name = format_ident!("expand");
        let func = func.to_tokens_mut();

        let out = quote! {
            #vis mod #name {
                use super::*;

                #[allow(unused, clippy::all)]
                #func

                #kernel
                #launch
                #launch_unchecked
                #dummy
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
                    __cube_count: #cube_count,
                    __cube_dim: #cube_dim,
                    #(#args),*
                ) -> () {
                    #body
                    launcher.launch(__cube_count, __kernel, __client);
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
                    __cube_count: #cube_count,
                    __cube_dim: #cube_dim,
                    #(#args),*
                ) -> () {
                    #body
                    launcher.launch_unchecked(__cube_count, __kernel, __client);
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
        let kernel_generics = self.kernel_generics.split_for_impl();
        let kernel_generics = kernel_generics.1.as_turbofish();
        let comptime_args = self.comptime_params().map(|it| &it.name);
        let (compilation_args, args) = self.compilation_args();

        quote! {
            use #core_path::frontend::ArgSettings as _;

            #settings
            #compilation_args

            let __kernel = #kernel_name #kernel_generics::new(__settings, #args #(#comptime_args),*);

            let mut launcher = #kernel_launcher::<__R>::default();

            #(#registers)*
        }
    }

    fn configure_settings(&self) -> TokenStream {
        let kernel_settings = prelude_type("KernelSettings");

        quote! {
            let mut __settings = #kernel_settings::default().cube_dim(__cube_dim);
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
            let generics = &self.launch_generics;
            let (_, generic_names, _) = self.kernel_generics.split_for_impl();

            let settings = self.configure_settings();
            let kernel_name = self.kernel_name();
            let core_path = core_path();
            let comptime_args = self.launch_args();
            let comptime_names = self.comptime_params().map(|it| &it.name);
            let (compilation_args, args) = self.compilation_args();

            quote! {
                #[allow(clippy::too_many_arguments)]
                #[doc = #kernel_doc]
                pub fn create_dummy_kernel #generics(
                    __cube_count: #cube_count,
                    __cube_dim: #cube_dim,
                    #(#comptime_args),*
                ) -> #kernel_name #generic_names {
                    use #core_path::frontend::ArgSettings as _;

                    #settings
                    #compilation_args

                    #kernel_name::new(__settings, #args #(#comptime_names),*)
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
        self.func.sig.runtime_params()
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
}
