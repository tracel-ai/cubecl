use std::iter;

use ident_case::RenameRule;
use proc_macro2::TokenStream;
use quote::{format_ident, quote, quote_spanned, ToTokens};
use syn::{parse_quote, spanned::Spanned, visit_mut::VisitMut, Generics, Ident};

use crate::{
    core_type, ir_path, ir_type,
    parse::{
        kernel::{Kernel, KernelParam},
        StripBounds,
    },
    prefix_ir, prelude_type,
    scope::Context,
};

impl ToTokens for Kernel {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let vis = &self.visibility;
        let name = &self.name;
        let generics = &self.generics;
        let global_constants = Context::new(self.returns.clone(), self.args.is_launch())
            .current_scope()
            .generate_kernel_vars();
        let block = &self.block;
        let return_type = &self.returns;
        let args = &self.parameters;

        let expr = ir_type("Expr");
        let ir_path = ir_path();

        let launch = self.launch();
        let launch_unchecked = self.launch_unchecked();
        let dummy = self.create_dummy_kernel();
        let kernel = self.kernel_definition();
        let checks = self.check_args();

        let out = quote! {
            #vis mod #name {
                use super::*;
                use #ir_path::{ExpandExpr as _, PartialExpand as _};

                #[allow(unused, clippy::all)]
                pub fn expand #generics(#(#args),*) -> impl #expr<Output = #return_type> {
                    #(#global_constants)*
                    {
                        #block
                    }
                }

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

impl ToTokens for KernelParam {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let name = &self.name;
        let ty = &self.normalized_ty;
        let span = self.span;
        tokens.extend(quote_spanned![span=>
            #name: #ty
        ]);
    }
}

impl Kernel {
    fn launch(&self) -> TokenStream {
        if self.args.launch.is_present() {
            let compute_client = prelude_type("ComputeClient");
            let cube_count = prelude_type("CubeCount");
            let cube_dim = prelude_type("CubeDim");

            let kernel_doc = format!("Launch the kernel [{}()] on the given runtime", self.name);
            let generics = self.launch_generics();
            let args = self.launch_args();
            let mut expand_generics = self.generics.clone();
            StripBounds.visit_generics_mut(&mut expand_generics);

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

            let kernel_doc = format!("Launch the kernel [{}()] on the given runtime", self.name);
            let generics = self.launch_generics();
            let args = self.launch_args();
            let mut expand_generics = self.generics.clone();
            StripBounds.visit_generics_mut(&mut expand_generics);

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
        let builder = ir_type("KernelBuilder");

        let expand_inputs = self.parameters.iter().map(|it| &it.name);
        let registers = self.runtime_params().map(|arg| {
            let name = &arg.name;
            quote![#name.register(&mut launcher);]
        });

        let mut expand_generics = self.generics.clone();
        StripBounds.visit_generics_mut(&mut expand_generics);
        let expand_generics =
            (!expand_generics.params.is_empty()).then(|| quote![::#expand_generics]);

        let settings = self.configure_settings();
        let io_mappings = self.io_mappings();
        let kernel_name = self.kernel_name();
        let hash = self.comptime_hash();

        quote! {
            use ::cubecl_core::frontend::ArgSettings as _;
            use ::cubecl_core::new_ir::Expr as _;

            #settings
            #hash
            let __settings__ = __settings.clone();
            let __expand = move || {
                let mut __builder = #builder::default();
                #io_mappings
                let expansion = expand #expand_generics(#(#expand_inputs),*);
                __builder.apply_expansion(expansion.expression_untyped());
                __builder.build(__settings.clone())
            };
            let kernel = #kernel_name {
                settings: __settings__,
                definition: __expand,
                comptime_hash: __comptime_hash
            };
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

    fn io_mappings(&self) -> TokenStream {
        let launch_arg_expand = ir_type("LaunchArgExpand");
        let global_var = ir_type("GlobalVariable");

        let input_expands = self.runtime_inputs().enumerate().map(|(i, arg)| {
            let name = &arg.name;
            let ty = arg.ty_owned();
            quote![let #name = <#ty as #launch_arg_expand>::expand(&mut __builder, __settings.vectorization_input(#i));]
        });
        let input_fn_mappings = self.runtime_inputs().enumerate().map(|(i, arg)| {
            let name = &arg.name;
            quote! {
                #i => Box::new(#name)
            }
        });

        let mappings = quote! {
            for __mapping in __settings.mappings.iter() {
                __map_assign(__mapping.pos_input, __mapping.pos_output);
            }
        };
        let output_expands = self.runtime_outputs().enumerate().map(|(i, arg)| {
            let name = &arg.name;
            let ty = arg.ty_owned();
            quote! {
                let #name = #name.unwrap_or_else(|| <#ty as #launch_arg_expand>::expand_output(
                    &mut __builder, __settings.vectorization_output(#i)
                ));
            }
        });

        let output_declarations = self.runtime_outputs().map(|arg| {
            let name = &arg.name;
            let ty = arg.ty_owned();
            quote![let mut #name: Option<#global_var<#ty>> = None;]
        });

        let set_out_mappings = self.runtime_outputs().enumerate().map(|(i, arg)| {
            let name = &arg.name;
            quote! {
                #i => {
                    #name = Some(*__input.downcast().unwrap());
                }
            }
        });
        let map_input = quote! {
            #[allow(unreachable_code)]
            let mut __map_assign = |__in_pos: usize, __out_pos: usize| {
                let __input: Box<dyn ::core::any::Any> = match __in_pos {
                    #(#input_fn_mappings,)*
                    _ => unreachable!()
                };
                match __out_pos {
                    #(#set_out_mappings,)*
                    _ => unreachable!()
                }
            };
        };

        quote! {
            #(#input_expands)*
            #(#output_declarations)*
            #map_input
            #mappings
            #(#output_expands)*
        }
    }

    fn create_dummy_kernel(&self) -> TokenStream {
        if self.args.create_dummy_kernel.is_present() {
            let cube_count = prelude_type("CubeCount");
            let cube_dim = prelude_type("CubeDim");
            let builder = ir_type("KernelBuilder");
            let kernel = core_type("Kernel");

            let kernel_doc = format!("Launch the kernel [{}()] on the given runtime", self.name);
            let generics = self.launch_generics();
            let args = self.launch_args();
            let mut expand_generics = self.generics.clone();
            StripBounds.visit_generics_mut(&mut expand_generics);
            let expand_generics =
                (!expand_generics.params.is_empty()).then(|| quote![::#expand_generics]);
            let expand_inputs = self.parameters.iter().map(|it| &it.name);

            let settings = self.configure_settings();
            let io_mappings = self.io_mappings();
            let kernel_name = self.kernel_name();
            let hash = self.comptime_hash();

            quote! {
                #[allow(clippy::too_many_arguments)]
                #[doc = #kernel_doc]
                pub fn create_dummy_kernel #generics(
                    __cube_count: #cube_count<__R::Server>,
                    __cube_dim: #cube_dim,
                    #(#args),*
                ) -> impl #kernel {
                    use ::cubecl_core::frontend::ArgSettings as _;
                    use ::cubecl_core::new_ir::Expr as _;

                    #settings
                    #hash
                    let __settings__ = __settings.clone();
                    let __expand = move || {
                        let mut __builder = #builder::default();
                        #io_mappings
                        let expansion = expand #expand_generics(#(#expand_inputs),*);
                        __builder.apply_expansion(expansion.expression_untyped());
                        __builder.build(__settings.clone())
                    };
                    #kernel_name {
                        settings: __settings__,
                        definition: __expand,
                        comptime_hash: __comptime_hash
                    }
                }
            }
        } else {
            TokenStream::new()
        }
    }

    fn runtime_inputs(&self) -> impl Iterator<Item = &KernelParam> {
        self.runtime_params().filter(|it| !it.is_mut)
    }

    fn runtime_outputs(&self) -> impl Iterator<Item = &KernelParam> {
        self.runtime_params().filter(|it| it.is_mut)
    }

    fn runtime_params(&self) -> impl Iterator<Item = &KernelParam> {
        self.parameters.iter().filter(|it| !it.is_const)
    }

    fn launch_generics(&self) -> Generics {
        let mut generics = self.generics.clone();
        let runtime = prelude_type("Runtime");
        generics.params = iter::once(parse_quote!['kernel])
            .chain(generics.params)
            .chain(iter::once(parse_quote![__R: #runtime]))
            .collect();
        generics
    }

    fn launch_args(&self) -> Vec<KernelParam> {
        let mut args = self.parameters.clone();
        let runtime_arg = ir_type("RuntimeArg");
        for arg in args.iter_mut().filter(|it| !it.is_const) {
            let ty = arg.ty_owned();
            arg.normalized_ty = parse_quote![#runtime_arg<'kernel, #ty, __R>];
        }
        args
    }

    fn kernel_name(&self) -> Ident {
        let kernel_name = RenameRule::PascalCase.apply_to_field(self.name.to_string());
        format_ident!("{kernel_name}")
    }

    fn comptime_hash(&self) -> TokenStream {
        let comptime_arg_hashes = self.parameters.iter().filter(|it| it.is_const).map(|arg| {
            let name = &arg.name;
            quote![::core::hash::Hash::hash(&#name, &mut __hasher);]
        });
        quote! {
            let __comptime_hash = {
                let mut __hasher = ::std::hash::DefaultHasher::new();
                #(#comptime_arg_hashes)*
                ::core::hash::Hasher::finish(&__hasher)
            };
        }
    }

    fn kernel_definition(&self) -> TokenStream {
        if self.args.is_launch() {
            let kernel = core_type("Kernel");
            let kernel_settings = prelude_type("KernelSettings");
            let kernel_definition: syn::Path = prelude_type("KernelDefinition");
            let kernel_id = core_type("KernelId");

            let kernel_name = self.kernel_name();
            let kernel_doc = format!("{} Kernel", self.name);

            quote! {
                #[doc = #kernel_doc]
                pub struct #kernel_name<F: Fn() -> #kernel_definition + Send + Sync + 'static> {
                    settings: #kernel_settings,
                    definition: F,
                    comptime_hash: u64
                }

                impl<F: Fn() -> #kernel_definition + Send + Sync + 'static> #kernel for #kernel_name<F> {
                    fn define(&self) -> #kernel_definition {
                        (self.definition)()
                    }

                    fn id(&self) -> #kernel_id {
                        #kernel_id::new::<Self>().info((self.settings.clone(), self.comptime_hash))
                    }
                }
            }
        } else {
            TokenStream::new()
        }
    }

    fn check_args(&self) -> TokenStream {
        if self.args.is_launch() {
            let generics = &self.generics;

            let input_checks = self
                .parameters
                .iter()
                // Const can be anything as long as the accessed fields are cube types, since the access
                // gets resolved at expansion time and collapsed into a literal in the kernel
                .filter(|arg| !arg.is_const)
                .map(|arg| {
                    let span = arg.ty.span();
                    let check = prefix_ir(format_ident!("assert_valid_type"));
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
