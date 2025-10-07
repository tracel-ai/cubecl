use darling::usage::{CollectLifetimes as _, CollectTypeParams as _, GenericsExt as _, Purpose};
use proc_macro2::TokenStream;
use quote::{ToTokens, format_ident, quote, quote_spanned};
use syn::{Ident, TypeParamBound};

use crate::{
    parse::kernel::{
        KernelBody, KernelFn, KernelParam, KernelReturns, KernelSignature, Launch, strip_ref,
    },
    paths::{frontend_type, prelude_path, prelude_type},
};

impl KernelFn {
    pub fn to_tokens_mut(&mut self) -> TokenStream {
        let prelude_path = prelude_path();

        let vis = &self.vis;
        let sig = &self.sig;
        let body = match &self.body {
            KernelBody::Block(block) => &block.to_tokens(&mut self.context),
            KernelBody::Verbatim(tokens) => tokens,
        };
        let name = &self.full_name;

        let (debug_source, debug_params) = if cfg!(debug_symbols) || self.debug_symbols {
            let debug_source = frontend_type("debug_source_expand");
            let cube_debug = frontend_type("CubeDebug");
            let src_file = self.src_file.as_ref().map(|file| file.value());
            let src_file = src_file.or_else(|| {
                let span: proc_macro::Span = self.span.unwrap();
                let source_path = span.local_file();
                let source_file = source_path.as_ref().and_then(|path| path.file_name());
                source_file.map(|file| file.to_string_lossy().into())
            });
            let source_text = match src_file {
                Some(file) => quote![include_str!(#file)],
                None => quote![""],
            };

            let debug_source = quote_spanned! {self.span=>
                #debug_source(scope, #name, file!(), #source_text, line!(), column!())
            };
            let debug_params = sig
                .runtime_params()
                .map(|it| &it.name)
                .map(|name| {
                    let name_str = name.to_string();
                    quote! [#cube_debug::set_debug_name(&#name, scope, #name_str);]
                })
                .collect();
            (debug_source, debug_params)
        } else {
            (TokenStream::new(), Vec::new())
        };

        let out = quote! {
            #vis #sig {
                #debug_source;
                #(#debug_params)*
                use #prelude_path::IntoRuntime as _;

                #body
            }
        };

        out
    }
}

impl ToTokens for KernelSignature {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let scope = prelude_type("Scope");
        let cube_type = prelude_type("CubeType");

        let name = &self.name;
        let generics = &self.generics;
        let return_type = match &self.returns {
            KernelReturns::ExpandType(ty) => {
                let mut is_mut = false;
                let mut is_ref = false;
                let ty = strip_ref(ty.clone(), &mut is_ref, &mut is_mut);
                quote![<#ty as #cube_type>::ExpandType]
            }
            KernelReturns::Plain(ty) => quote![#ty],
        };
        let out = if let Some(receiver) = &self.receiver_arg {
            let args = self.parameters.iter().skip(1);

            quote! {
                fn #name #generics(
                    #receiver,
                    scope: &mut #scope,
                    #(#args),*
                ) -> #return_type
            }
        } else {
            let args = &self.parameters;
            quote! {
                fn #name #generics(
                    scope: &mut #scope,
                    #(#args),*
                ) -> #return_type
            }
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

    pub fn compilation_args_def(&self) -> (Vec<TokenStream>, Vec<Ident>) {
        let mut tokens = Vec::new();
        let mut args = Vec::new();
        let launch_arg = prelude_type("LaunchArg");

        self.runtime_inputs().for_each(|input| {
            let ty = &input.ty_owned();
            let name = &input.name;

            tokens.push(quote! {
                #name: <#ty as #launch_arg>::CompilationArg
            });
            args.push(name.clone());
        });

        self.runtime_outputs().for_each(|output| {
            let ty = &output.ty_owned();
            let name = &output.name;

            tokens.push(quote! {
                #name: <#ty as #launch_arg>::CompilationArg
            });
            args.push(name.clone());
        });

        (tokens, args)
    }

    pub fn compilation_args(&self) -> (TokenStream, TokenStream) {
        let launch_arg = prelude_type("LaunchArg");
        let mut defined = quote! {};
        let mut args = quote! {};

        self.runtime_inputs().enumerate().for_each(|(i, input)| {
            let ty = &input.ty_owned();
            let ident = &input.name;
            let var = Ident::new(format!("input_arg_{i}").as_str(), ident.span());

            args.extend(quote! {#var,});
            defined.extend(quote! {
                let #var = <#ty as #launch_arg>::compilation_arg::<__R>(&#ident);
            });
        });
        self.runtime_outputs().enumerate().for_each(|(i, output)| {
            let ty = &output.ty_owned();
            let ident = &output.name;
            let var = Ident::new(format!("output_arg_{i}").as_str(), ident.span());

            args.extend(quote! {#var,});
            defined.extend(quote! {
                let #var = <#ty as #launch_arg>::compilation_arg::<__R>(&#ident);
            });
        });

        (
            quote! {
                #defined
            },
            args,
        )
    }

    pub fn io_mappings(&self) -> TokenStream {
        let launch_arg = prelude_type("LaunchArg");
        let mut define = quote! {};

        let expand_fn = |ident, expand_name, ty| {
            let ty = self.analysis.process_ty(&ty);

            quote! {
                let #ident =  <#ty as #launch_arg>::#expand_name(&self.#ident.dynamic_cast(), &mut builder);
            }
        };
        for param in self.runtime_params() {
            let expand_name = match param.is_mut {
                true => format_ident!("expand_output"),
                false => format_ident!("expand"),
            };
            define.extend(expand_fn(&param.name, expand_name, param.ty_owned()));
        }

        quote! {
            #define
        }
    }

    fn define_body(&self) -> TokenStream {
        let kernel_builder = prelude_type("KernelBuilder");
        let io_map = self.io_mappings();
        let register_type = self.analysis.register_types();
        let runtime_args = self.runtime_params().map(|it| &it.name);
        let comptime_args = self.comptime_params().map(|it| &it.name);
        let generics = self.analysis.process_generics(&self.func.sig.generics);

        quote! {
            let mut builder = #kernel_builder::default();
            builder.runtime_properties(__R::target_properties());
            #register_type
            #io_map
            expand #generics(&mut builder.scope, #(#runtime_args.clone(),)* #(self.#comptime_args.clone()),*);
            builder.build(self.settings.clone())
        }
    }

    /// Returns the kernel entrypoint name.
    /// Appropriate for usage in source code such as naming the CUDA or WGSL
    /// entrypoint.
    ///
    /// For example a kernel:
    /// ```text
    /// #[cube(launch)]
    /// fn my_kernel(input: &Array<f32>, output: &mut Array<f32>) {}
    /// ```
    /// would produce the name `my_kernel`.
    ///
    /// If a generic has the `Float` or `Numeric` bound the kernel also has a
    /// suffix with the name of that type in use:
    /// ```text
    /// fn my_kernel<F: Float>(input: &Array<F>, output: &mut Array<F>) {}
    /// ```
    /// now produces the name `my_kernel_f16` or `my_kernel_f32` etc. depending
    /// on which variant of the kernel is launched by the user.
    ///
    /// If a kernel has several matching bounds they are appended as suffixes in
    /// order.
    fn kernel_entrypoint_name(&self) -> TokenStream {
        // This base name is always used; a suffix might be added
        // based on generics.
        let base_name = self.func.sig.name.to_string();

        let generics = &self.kernel_generics;
        let suffix_producing_bounds = [format_ident!("Float"), format_ident!("Numeric")];

        let mut matching_generics = vec![];

        // Inspect all generics for the bounds of interest in order to
        // determine if a suffix should be added
        for ty in generics.type_params() {
            for bound in &ty.bounds {
                let TypeParamBound::Trait(t) = bound else {
                    continue;
                };

                // Using last should account for the bounds such as `Float` but also
                // `some::prefix::Float`
                let Some(generic_trailing) = t.path.segments.last() else {
                    continue;
                };

                // If we find some type parameter with `Float` as a bound,
                // add a suffix based on a shortened version of the
                // type name
                if suffix_producing_bounds.contains(&generic_trailing.ident) {
                    // E.g. the `F` in `F: Float` or `N` in `N: Numeric`
                    matching_generics.push(ty.ident.clone());
                    continue;
                }
            }
        }

        if matching_generics.is_empty() {
            quote! {
                #base_name
            }
        } else {
            quote! (
                {
                    // Go from type names such as `half::f16` to `f16` etc.
                    let shorten = |p: &'static str| {
                        if let Some((_, last)) = p.rsplit_once("::") {
                            last
                        } else {
                            p
                        }
                    };

                    let mut name = format!("{}", #base_name);

                    #( {
                        let type_name = shorten(core::any::type_name::< #matching_generics >());
                        name.push_str(&format!("_{type_name}"));
                    })*

                    name
                }
            )
        }
    }

    pub fn kernel_definition(&self) -> TokenStream {
        if self.args.is_launch() {
            let kernel_metadata = prelude_type("KernelMetadata");
            let cube_kernel = prelude_type("CubeKernel");
            let kernel_settings = prelude_type("KernelSettings");
            let kernel_definition: syn::Path = prelude_type("KernelDefinition");
            let kernel_id = prelude_type("KernelId");

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
            let phantom_data_init = phantom_data
                .as_ref()
                .map(|_| quote![__ty: ::core::marker::PhantomData]);
            let (compilation_args, args) = self.compilation_args_def();
            let info = param_names.clone().into_iter().chain(args.clone());

            let kernel_source_name = self.kernel_entrypoint_name();
            let mut settings = quote![settings.kernel_name(#kernel_source_name)];
            if self.args.debug_symbols.is_present() {
                settings.extend(quote![.debug_symbols()]);
            }
            if let Some(mode) = &self.args.fast_math {
                settings.extend(quote![.fp_math_mode((#mode).into())]);
            }
            if let Some(cluster_dim) = &self.args.cluster_dim {
                settings.extend(quote![.cluster_dim(#cluster_dim)]);
            }

            quote! {
                #[doc = #kernel_doc]
                pub struct #kernel_name #generics #where_clause {
                    settings: #kernel_settings,
                    #(#compilation_args,)*
                    #(#const_params,)*
                    #phantom_data
                }

                #[allow(clippy::too_many_arguments)]
                impl #generics #kernel_name #generic_names #where_clause {
                    pub fn new(settings: #kernel_settings, #(#compilation_args,)* #(#const_params),*) -> Self {
                        Self {
                            settings: #settings,
                            #(#args,)*
                            #(#param_names,)*
                            #phantom_data_init
                        }
                    }
                }

                impl #generics #kernel_metadata for #kernel_name #generic_names #where_clause {
                    fn id(&self) -> #kernel_id {
                        // We don't use any other kernel settings with the macro.
                        let cube_dim = self.settings.cube_dim.clone();
                        #kernel_id::new::<Self>().info((cube_dim, #(self.#info.clone()),* ))
                    }
                }

                impl #generics #cube_kernel for #kernel_name #generic_names #where_clause {
                    fn define(&self) -> #kernel_definition {
                        #define
                    }
                }
            }
        } else {
            TokenStream::new()
        }
    }
}
