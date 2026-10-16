//! A `harness` function for every pure cube function: one whose parameters
//! and return value are all scalars. The harness launches the function on a
//! single unit of the given runtime and hands its result back to the host, so
//! a test can call `f::harness::<R>(1)` instead of writing a kernel, buffers
//! and a read back by hand.
//!
//! Only emitted when the `harness` feature is enabled.

use darling::util::Flag;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{GenericParam, Ident, ItemFn, Type, TypeParamBound, parse_quote};

use crate::{
    parse::{
        kernel::{KernelArgs, KernelBody, Launch},
        signature::KernelReturns,
    },
    paths::{core_type, prelude_type},
};

/// Scalars that can be passed to a kernel at runtime and read back from a
/// buffer on the host.
const RUNTIME_SCALARS: &[&str] = &[
    "u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "f16", "bf16", "f32", "f64",
];

/// Scalars that are only valid as comptime arguments, since they aren't plain
/// old data the host can read back.
const COMPTIME_SCALARS: &[&str] = &["bool", "usize", "isize"];

/// The generic bounds a harness can be instantiated over.
const GENERIC_BOUNDS: &[&str] = &["Float", "Int", "Numeric"];

/// How a value crosses from the host to the harness kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HarnessValue {
    /// A concrete scalar, passed as is.
    Scalar(Type),
    /// A generic scalar, passed as an `InputScalar` and read back as the
    /// generic type in the kernel.
    Generic(Ident),
    /// A comptime argument, passed as is.
    Comptime(Type),
}

pub struct HarnessPlan {
    pub generics: Vec<(Ident, Ident)>,
    pub params: Vec<(Ident, HarnessValue)>,
    pub returns: HarnessValue,
}

impl Launch {
    pub fn harness(&self) -> TokenStream {
        if !cfg!(feature = "harness") {
            return TokenStream::new();
        }

        match self.harness_plan() {
            Some(plan) => self
                .harness_tokens(&plan)
                .unwrap_or_else(|err| err.into_compile_error()),
            None => TokenStream::new(),
        }
    }

    /// The plan for this function's harness, or `None` when the function
    /// isn't pure.
    pub fn harness_plan(&self) -> Option<HarnessPlan> {
        let sig = &self.func.sig;

        if self.args.is_launch()
            || self.args.expand_only.is_present()
            || self.func.context.is_intrinsic
            || sig.receiver_arg.is_some()
            || sig.scope_lifetime.is_some()
            || sig.generics.where_clause.is_some()
            || !matches!(self.func.body, KernelBody::Block(_))
        {
            return None;
        }

        let mut generics = Vec::new();
        for param in sig.generics.params.iter() {
            let GenericParam::Type(param) = param else {
                return None;
            };
            if !param.attrs.is_empty() || param.default.is_some() || param.bounds.len() != 1 {
                return None;
            }
            let Some(TypeParamBound::Trait(bound)) = param.bounds.first() else {
                return None;
            };
            let bound = bound.path.get_ident()?;
            if !GENERIC_BOUNDS.contains(&bound.to_string().as_str()) {
                return None;
            }
            generics.push((param.ident.clone(), bound.clone()));
        }

        let classify = |ty: &Type| -> Option<HarnessValue> {
            let Type::Path(path) = ty else {
                return None;
            };
            if path.qself.is_some() {
                return None;
            }
            if let Some(ident) = path.path.get_ident()
                && generics.iter().any(|(generic, _)| generic == ident)
            {
                return Some(HarnessValue::Generic(ident.clone()));
            }
            let last = path.path.segments.last()?;
            (last.arguments.is_none() && RUNTIME_SCALARS.contains(&last.ident.to_string().as_str()))
                .then(|| HarnessValue::Scalar(ty.clone()))
        };

        let mut params = Vec::new();
        for param in sig.parameters.iter() {
            if !param.defines.is_empty() {
                return None;
            }
            let value = if param.is_const {
                let ident = match &param.ty {
                    Type::Path(path) if path.qself.is_none() => path.path.get_ident(),
                    _ => None,
                }?;
                let name = ident.to_string();
                if !RUNTIME_SCALARS.contains(&name.as_str())
                    && !COMPTIME_SCALARS.contains(&name.as_str())
                {
                    return None;
                }
                HarnessValue::Comptime(param.ty.clone())
            } else {
                classify(&param.ty)?
            };
            params.push((param.name.clone(), value));
        }

        let KernelReturns::ExpandType(returns) = &sig.returns else {
            return None;
        };
        let returns = classify(returns)?;

        Some(HarnessPlan {
            generics,
            params,
            returns,
        })
    }

    fn harness_tokens(&self, plan: &HarnessPlan) -> syn::Result<TokenStream> {
        let input_scalar = prelude_type("InputScalar");
        let cube_element = prelude_type("CubeElement");
        let buffer_arg = prelude_type("BufferArg");
        let cube_count = prelude_type("CubeCount");
        let cube_dim = prelude_type("CubeDim");
        let runtime = core_type("__private");
        let runtime = quote![#runtime::Runtime];

        let name = &self.func.sig.name;
        let kernel_name = format_ident!("{name}_harness");
        let generics = &self.func.sig.generics;
        let generic_names = plan.generics.iter().map(|(ident, _)| ident);
        let turbofish = match plan.generics.is_empty() {
            true => quote![],
            false => quote![::<#(#generic_names),*>],
        };
        let ret = match &plan.returns {
            HarnessValue::Scalar(ty) | HarnessValue::Comptime(ty) => quote![#ty],
            HarnessValue::Generic(ident) => quote![#ident],
        };

        // The kernel calls the function through a path the cube parser turns
        // into a call to this module's `expand`. `super::expand` can't be used
        // directly, since the parser would append another `::expand`.
        let mut kernel_params = Vec::new();
        let mut call_args = Vec::new();
        for (param, value) in plan.params.iter() {
            match value {
                HarnessValue::Scalar(ty) => {
                    kernel_params.push(quote![#param: #ty]);
                    call_args.push(quote![#param]);
                }
                HarnessValue::Generic(ident) => {
                    kernel_params.push(quote![#param: #input_scalar]);
                    call_args.push(quote![#param.get::<#ident>()]);
                }
                HarnessValue::Comptime(ty) => {
                    kernel_params.push(quote![#[comptime] #param: #ty]);
                    call_args.push(quote![#param]);
                }
            }
        }
        let kernel: ItemFn = parse_quote! {
            fn #kernel_name #generics(#(#kernel_params,)* __harness_out: &mut [#ret]) {
                __harness_out[0] = __harness_call #turbofish(#(#call_args),*);
            }
        };
        let kernel_args = KernelArgs {
            launch: Flag::present(),
            debug_symbols: self.args.debug_symbols,
            no_debug_symbols: self.args.no_debug_symbols,
            src_file: self.args.src_file.clone(),
            ..Default::default()
        };
        let kernel = Launch::from_item_fn(kernel, kernel_args)?;

        let harness_generics = plan
            .generics
            .iter()
            .map(|(ident, bound)| quote![#ident: #bound + #cube_element]);
        let harness_params = plan.params.iter().map(|(param, value)| match value {
            HarnessValue::Scalar(ty) | HarnessValue::Comptime(ty) => quote![#param: #ty],
            HarnessValue::Generic(ident) => quote![#param: #ident],
        });
        let launch_args = plan.params.iter().map(|(param, value)| match value {
            HarnessValue::Generic(ident) => {
                quote![#input_scalar::new(#param, <#ident as #cube_element>::cube_type())]
            }
            _ => quote![#param],
        });
        let name_str = name.to_string();
        let doc = format!(
            "Run [{name_str}()] on a single unit of the given runtime and return its result.\n\n\
             # Panics\n\n\
             If the kernel fails to compile or run."
        );

        Ok(quote! {
            #[doc(hidden)]
            pub mod __harness_call {
                pub use super::expand;
            }

            #kernel

            #[allow(dead_code, clippy::too_many_arguments)]
            #[doc = #doc]
            pub fn harness<__R: #runtime, #(#harness_generics),*>(#(#harness_params),*) -> #ret {
                let __client = <__R as #runtime>::client(&::core::default::Default::default());
                let __out = __client.empty(::core::mem::size_of::<#ret>());
                #kernel_name::launch #turbofish(
                    &__client,
                    #cube_count::new_single(),
                    #cube_dim::new_single(),
                    #(#launch_args,)*
                    // SAFETY: `__out` holds exactly one element.
                    unsafe { #buffer_arg::from_raw_parts(__out.clone(), 1) },
                );
                let __bytes = __client
                    .read_one(__out)
                    .unwrap_or_else(|err| panic!("harness for `{}` failed: {err}", #name_str));
                <#ret as #cube_element>::from_bytes(&__bytes)[0]
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::kernel::from_tokens;
    use quote::ToTokens;

    fn plan(item: ItemFn) -> Option<HarnessPlan> {
        Launch::from_item_fn(item, KernelArgs::default())
            .unwrap()
            .harness_plan()
    }

    #[test]
    fn scalar_functions_are_pure() {
        let plan = plan(parse_quote! {
            fn f(x: u32, y: half::f16, #[comptime] n: usize, #[comptime] b: bool) -> f32 { x }
        })
        .unwrap();

        assert!(plan.generics.is_empty());
        assert_eq!(plan.returns, HarnessValue::Scalar(parse_quote!(f32)));
        assert_eq!(
            plan.params
                .iter()
                .map(|(_, v)| v.clone())
                .collect::<Vec<_>>(),
            vec![
                HarnessValue::Scalar(parse_quote!(u32)),
                HarnessValue::Scalar(parse_quote!(half::f16)),
                HarnessValue::Comptime(parse_quote!(usize)),
                HarnessValue::Comptime(parse_quote!(bool)),
            ]
        );
    }

    #[test]
    fn generic_scalar_functions_are_pure() {
        let plan = plan(parse_quote! {
            fn f<F: Float, I: Int>(x: F, y: I, z: u32) -> F { x }
        })
        .unwrap();

        assert_eq!(plan.generics.len(), 2);
        assert_eq!(plan.returns, HarnessValue::Generic(parse_quote!(F)));
        assert_eq!(plan.params[1].1, HarnessValue::Generic(parse_quote!(I)));
    }

    #[test]
    fn impure_functions_have_no_harness() {
        let impure: Vec<ItemFn> = vec![
            parse_quote! { fn f(x: u32) {} },
            parse_quote! { fn f(x: bool) -> u32 { 0 } },
            parse_quote! { fn f(x: u32) -> bool { true } },
            parse_quote! { fn f(x: usize) -> u32 { 0 } },
            parse_quote! { fn f(x: &u32) -> u32 { 0 } },
            parse_quote! { fn f(x: &[u32]) -> u32 { 0 } },
            parse_quote! { fn f(x: Vector<f32, N>) -> f32 { 0 } },
            parse_quote! { fn f<N: Size>(x: u32) -> u32 { 0 } },
            parse_quote! { fn f<F: Float + CubeElement>(x: F) -> F { x } },
            parse_quote! { fn f<F: Float>(x: F) -> F where F: Copy { x } },
            parse_quote! { fn f<T: CubePrimitive>(x: T) -> T { x } },
            parse_quote! { fn f(#[comptime] c: MyConfig) -> u32 { 0 } },
        ];
        for item in impure {
            let source = item.to_token_stream().to_string();
            assert!(plan(item).is_none(), "`{source}` shouldn't be pure");
        }

        let launch = Launch::from_item_fn(
            parse_quote! { fn f(x: u32, out: &mut [u32]) {} },
            from_tokens(quote![launch]).unwrap(),
        )
        .unwrap();
        assert!(launch.harness_plan().is_none());
    }

    // The harness kernel goes through the same expansion as the function, so
    // it must be as deterministic.
    #[test]
    fn harness_expansion_is_deterministic() {
        let item: ItemFn = parse_quote! {
            fn f<F: Float, I: Int>(x: F, y: I, z: u32, #[comptime] c: u32) -> F { x }
        };
        let expand = || {
            let launch = Launch::from_item_fn(item.clone(), KernelArgs::default()).unwrap();
            let plan = launch.harness_plan().unwrap();
            launch.harness_tokens(&plan).unwrap().to_string()
        };

        let first = expand();
        for _ in 0..16 {
            assert_eq!(expand(), first);
        }
    }
}
