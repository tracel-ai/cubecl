use ident_case::RenameRule;
use proc_macro2::TokenStream;
use quote::{ToTokens, format_ident, quote};
use syn::{GenericParam, Generics, Ident, parse_quote};

use crate::{
    parse::{
        kernel::{AddressType, GenericArg, Launch, anon_lifetime_to_static, strip_ref},
        signature::KernelParam,
    },
    paths::{core_type, prelude_type},
};

impl ToTokens for Launch {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let vis = &self.vis;

        let name = &self.func.sig.name;
        let launch = self.launch();
        let launch_unchecked = self.launch_unchecked();
        let aliases = self.create_type_alias();
        let dummy = self.create_dummy_kernel();
        let kernel = self.kernel_definition();
        let mut func = self.func.clone();
        func.sig.name = format_ident!("expand");
        let func = func.to_tokens_mut();

        let out = quote! {
            #vis mod #name {
                use super::*;

                #aliases

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
            let address_type = prelude_type("AddressType");

            let kernel_doc = format!(
                "Launch the kernel [{}()] on the given runtime",
                self.func.sig.name
            );
            let generics = &self.launch_generics;
            let args = self.launch_args();
            let body = self.launch_body();

            let address_type = match self.args.address_type {
                AddressType::Dynamic => quote![__address_type: #address_type,],
                _ => quote![],
            };

            quote! {
                #[allow(clippy::too_many_arguments)]
                #[doc = #kernel_doc]
                pub fn launch #generics(
                    __client: &#compute_client<__R>,
                    __cube_count: #cube_count,
                    __cube_dim: #cube_dim,
                    #address_type
                    #(#args),*
                ) {
                    #body
                    launcher.launch(__cube_count, __kernel, __client)
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
            let address_type = prelude_type("AddressType");

            let kernel_doc = format!(
                "Launch the kernel [{}()] on the given runtime without bound checks.\n\n\
                 # Safety\n\n\
                 The kernel must not:\n\
                 - Contain any out of bounds reads or writes. Doing so is immediate UB.\n\
                 - Contain any loops that never terminate. These may be optimized away entirely or cause\n\
                   other unpredictable behaviour.",
                self.func.sig.name
            );
            let generics = &self.kernel_generics;
            let args = self.launch_args();
            let body = self.launch_body();

            let address_type = match self.args.address_type {
                AddressType::Dynamic => quote![__address_type: #address_type,],
                _ => quote![],
            };

            quote! {
                #[allow(clippy::too_many_arguments)]
                #[doc = #kernel_doc]
                pub unsafe fn launch_unchecked #generics(
                    __client: &#compute_client<__R>,
                    __cube_count: #cube_count,
                    __cube_dim: #cube_dim,
                    #address_type
                    #(#args),*
                ) {
                    #body
                    launcher.launch_unchecked(__cube_count, __kernel, __client)
                }
            }
        } else {
            TokenStream::new()
        }
    }

    fn launch_body(&self) -> TokenStream {
        let kernel_launcher = prelude_type("KernelLauncher");
        let launch_context = prelude_type("LaunchContext");

        let mappings = self.func.sig.define_mappings();
        let generic_registers =
            self.func
                .analysis
                .register_types(mappings, quote![scope], false, true);

        let settings = self.configure_settings(true);
        let auto_address_type = self.configure_auto_address_type();
        let kernel_name = self.kernel_name();
        let kernel_generics = self.kernel_call_generics();
        let kernel_generics = kernel_generics.split_for_impl();
        let kernel_generics = kernel_generics.1.as_turbofish();
        let comptime_args = self.comptime_params().map(|it| &it.name);
        let (registers, args) = self.arg_registers();

        quote! {
            let __launch_context = #launch_context::new();
            __launch_context.with_scope(|scope| {
                scope.device_properties(__client.properties());
                #generic_registers
            });

            #auto_address_type
            #settings

            let mut launcher = #kernel_launcher::<__R>::new_with_context(
                __settings.clone(),
                __launch_context,
            );
            #registers
            let __kernel = #kernel_name #kernel_generics::new(__settings, __client.clone(), #args #(#comptime_args),*);
        }
    }

    fn configure_settings(&self, auto_resolved: bool) -> TokenStream {
        let kernel_settings = prelude_type("KernelSettings");
        let addr_ty = prelude_type("AddressType");
        let address_type = match self.args.address_type {
            AddressType::U32 => quote![#addr_ty::U32],
            AddressType::U64 => quote![#addr_ty::U64],
            AddressType::Dynamic => quote![__address_type],
            AddressType::Auto if auto_resolved => quote![__address_type],
            AddressType::Auto => quote![#addr_ty::U32],
        };

        quote! {
            let __settings = #kernel_settings::default()
                .cube_dim(__cube_dim).address_type(#address_type);
        }
    }

    fn configure_auto_address_type(&self) -> TokenStream {
        if self.args.address_type != AddressType::Auto {
            return TokenStream::new();
        }

        let launch_arg = prelude_type("LaunchArg");
        let address_type = prelude_type("AddressType");
        let required_address_types = self.runtime_params().map(|input| {
            let ty = strip_ref(input.ty.clone());
            let ty = anon_lifetime_to_static(ty);
            let name = &input.name;
            quote! {
                __address_type = __address_type.max(
                    <#ty as #launch_arg>::required_address_type::<__R>(&#name, scope)
                );
            }
        });

        quote! {
            let __address_type = __launch_context.with_scope(|scope| {
                let mut __address_type = #address_type::U32;
                #(#required_address_types)*
                __address_type
            });
            assert!(
                __client.properties().supports_address(__address_type),
                "The automatically selected address type `{}` isn't supported by this device",
                __address_type,
            );
        }
    }

    fn create_type_alias(&self) -> TokenStream {
        let mut aliases = quote! {};
        if !self.func.args.explicit_define.is_present() {
            for (
                name,
                GenericArg {
                    expand_ty,
                    marker_ty,
                    ..
                },
            ) in self.func.analysis.map.iter()
            {
                aliases.extend(quote! {
                    pub struct #marker_ty;
                    /// Type to be used as a generic for launch kernel argument.
                    pub type #name = #expand_ty;
                });
            }
        }

        aliases
    }
    fn create_dummy_kernel(&self) -> TokenStream {
        if self.args.create_dummy_kernel.is_present() {
            let cube_count = prelude_type("CubeCount");
            let cube_dim = prelude_type("CubeDim");
            let address_type = prelude_type("AddressType");

            let kernel_doc = format!(
                "Launch the kernel [{}()] on the given runtime",
                self.func.sig.name
            );
            let generics = &self.kernel_generics;
            let (_, generic_names, _) = self.kernel_generics.split_for_impl();

            let settings = self.configure_settings(false);
            let kernel_name = self.kernel_name();
            let comptime_args = self.launch_args();
            let comptime_names = self.comptime_params().map(|it| &it.name);
            let (compilation_args, args) = self.arg_registers();

            let address_type = match self.args.address_type {
                AddressType::Dynamic => quote![__address_type: #address_type,],
                _ => quote![],
            };

            quote! {
                #[allow(clippy::too_many_arguments)]
                #[doc = #kernel_doc]
                pub fn create_dummy_kernel #generics(
                    __cube_count: #cube_count,
                    __cube_dim: #cube_dim,
                    #address_type
                    #(#comptime_args),*
                ) -> #kernel_name #generic_names {
                    #settings
                    #compilation_args

                    #kernel_name::new(__settings, #args #(#comptime_names),*)
                }
            }
        } else {
            TokenStream::new()
        }
    }

    pub fn runtime_params(&self) -> impl Iterator<Item = &KernelParam> {
        self.func.sig.runtime_params()
    }

    fn launch_args(&self) -> Vec<KernelParam> {
        let mut args = self.func.sig.parameters.clone();
        let runtime_arg = core_type("RuntimeArg");
        for arg in args.iter_mut().filter(|it| !it.is_const) {
            let ty = strip_ref(arg.ty.clone());
            let ty = anon_lifetime_to_static(ty);
            arg.normalized_ty = parse_quote![#runtime_arg<#ty, __R>];
            arg.mutability = None;
        }
        args
    }

    pub fn kernel_name(&self) -> Ident {
        let kernel_name = RenameRule::PascalCase.apply_to_field(self.func.sig.name.to_string());
        format_ident!("{kernel_name}")
    }

    pub fn kernel_call_generics(&self) -> Generics {
        let mut generics = self.kernel_generics.clone();
        generics.params = generics
            .params
            .into_iter()
            .filter(|it| !matches!(it, GenericParam::Lifetime(..)))
            .collect();
        generics
    }

    pub fn comptime_params(&self) -> impl Iterator<Item = &KernelParam> {
        self.func
            .sig
            .parameters
            .iter()
            .filter(|param| param.is_const)
    }
}
