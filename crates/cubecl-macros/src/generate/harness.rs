//! A `harness` function for every cube function marked `#[cube(harness)]`,
//! whose parameters must all be scalars, slices or arrays of them, or cube
//! types, and that returns them or a tuple of them, or nothing when it writes
//! into a mutable slice. The harness launches the function on a single unit of
//! the given runtime and hands its result back to the host, so a test can call
//! `f::harness::<R>(1)` instead of writing a kernel, buffers and a read back by
//! hand.
//!
//! A slice is uploaded from a host slice of the same type, and a mutable one
//! is written back into it after the launch, as if the function ran on the
//! host. An `Array` is uploaded from a host slice too, and copied into an
//! array of its length in the kernel, since an array's length is comptime.
//!
//! A returned `Array` comes back as a `Vec`. Its length is comptime too, and
//! only known once the kernel is expanded, so a harness returning one expands
//! its kernel first to learn the length, and only then allocates its buffer.
//!
//! A function may also take and return cube types, which must derive
//! `CubeHarness`. A macro can't tell a cube type from any other path, so any
//! path that isn't a scalar is taken for one, and the bound on it reports a
//! type that isn't.
//!
//! The harness is only compiled in tests unless another gate is given, with
//! `harness(cfg(<predicate>))` or `harness(always)`. A function that isn't
//! harnessable is reported whatever the gate.

use darling::util::Flag;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    GenericArgument, GenericParam, Ident, ItemFn, PathArguments, Type, TypeParamBound,
    WherePredicate, parse_quote,
};

use crate::{
    parse::{
        harness::HarnessGate,
        kernel::{KernelArgs, KernelBody, Launch},
        signature::KernelReturns,
    },
    paths::{core_path, core_type, prelude_type},
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
    /// A cube type, passed as its launch argument.
    Cube(Type),
    /// A slice of scalars, uploaded to its own buffer, and written back to
    /// the host when it's mutable.
    Slice { elem: Type, mutable: bool },
    /// An array of scalars, uploaded from a host slice to its own buffer, and
    /// copied into an array of the slice's length in the kernel.
    Array(Type),
}

pub struct HarnessPlan {
    pub generics: Vec<(Ident, Ident)>,
    pub params: Vec<(Ident, HarnessValue)>,
    /// A value or a tuple of them, possibly nested, read back through its
    /// `HarnessOutput` impl.
    pub returns: Type,
}

/// Whether a return type holds an `Array`, whose length the harness only
/// learns by expanding its kernel.
fn returns_array(ty: &Type) -> bool {
    match ty {
        Type::Tuple(tuple) => tuple.elems.iter().any(returns_array),
        ty => array_elem(ty).is_some(),
    }
}

/// The element type of an `Array<T>`.
fn array_elem(ty: &Type) -> Option<&Type> {
    let Type::Path(path) = ty else {
        return None;
    };
    let last = path.path.segments.last()?;
    if path.qself.is_some() || last.ident != "Array" {
        return None;
    }
    let PathArguments::AngleBracketed(args) = &last.arguments else {
        return None;
    };
    match args.args.iter().collect::<Vec<_>>()[..] {
        [GenericArgument::Type(elem)] => Some(elem),
        _ => None,
    }
}

impl Launch {
    pub fn harness(&self) -> TokenStream {
        let Some(gate) = &self.args.harness else {
            return TokenStream::new();
        };

        match self.harness_plan() {
            Some(plan) => self
                .harness_tokens(&plan, gate)
                .unwrap_or_else(|err| err.into_compile_error()),
            None => syn::Error::new_spanned(
                &self.func.sig.name,
                "a harness can't be generated for this function. Every parameter must be one of:\n\
                 - a scalar, or a `#[comptime]` scalar\n\
                 - a generic scalar, with a single `Float`, `Int` or `Numeric` bound, inline or \
                 in a `where` clause\n\
                 - a slice `&[T]` or `&mut [T]` of scalars\n\
                 - an `Array<T>` of scalars\n\
                 - a cube type deriving `CubeHarness`\n\
                 and the function must return them, a tuple of them, or nothing when it writes \
                 into a mutable slice",
            )
            .into_compile_error(),
        }
    }

    /// The plan for this function's harness, or `None` when the function
    /// isn't harnessable.
    pub fn harness_plan(&self) -> Option<HarnessPlan> {
        let sig = &self.func.sig;

        if self.args.is_launch()
            || self.args.expand_only.is_present()
            || self.func.context.is_intrinsic
            || sig.receiver_arg.is_some()
            || sig.scope_lifetime.is_some()
            || !matches!(self.func.body, KernelBody::Block(_))
        {
            return None;
        }

        // `where F: Float` is the same as an inline `F: Float`, so a predicate
        // on a bare generic is folded into its bounds. No other predicate is
        // supported.
        let mut where_bounds = Vec::new();
        for predicate in sig.generics.where_clause.iter().flat_map(|w| &w.predicates) {
            let WherePredicate::Type(predicate) = predicate else {
                return None;
            };
            let Type::Path(ty) = &predicate.bounded_ty else {
                return None;
            };
            if predicate.lifetimes.is_some() || ty.qself.is_some() {
                return None;
            }
            where_bounds.push((ty.path.get_ident()?, &predicate.bounds));
        }
        let is_type_param = |ident: &Ident| {
            sig.generics
                .type_params()
                .any(|param| &param.ident == ident)
        };
        if !where_bounds.iter().all(|(ident, _)| is_type_param(ident)) {
            return None;
        }

        let mut generics = Vec::new();
        for param in sig.generics.params.iter() {
            let GenericParam::Type(param) = param else {
                return None;
            };
            let bounds = param
                .bounds
                .iter()
                .chain(
                    where_bounds
                        .iter()
                        .filter(|(ident, _)| *ident == &param.ident)
                        .flat_map(|(_, bounds)| bounds.iter()),
                )
                .collect::<Vec<_>>();
            if !param.attrs.is_empty() || param.default.is_some() {
                return None;
            }
            let [TypeParamBound::Trait(bound)] = bounds[..] else {
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
            // An array is a container, not a cube type, and is classified by
            // its element where it can cross.
            if array_elem(ty).is_some() {
                return None;
            }
            let last = path.path.segments.last()?;
            let name = last.ident.to_string();
            if last.arguments.is_none() && RUNTIME_SCALARS.contains(&name.as_str()) {
                return Some(HarnessValue::Scalar(ty.clone()));
            }
            // Caught here rather than as a cube type, for a clearer error.
            if COMPTIME_SCALARS.contains(&name.as_str()) {
                return None;
            }
            Some(HarnessValue::Cube(ty.clone()))
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
            } else if let Type::Reference(reference) = &param.ty {
                let Type::Slice(slice) = &*reference.elem else {
                    return None;
                };
                match classify(&slice.elem)? {
                    HarnessValue::Scalar(_) | HarnessValue::Generic(_) => HarnessValue::Slice {
                        elem: (*slice.elem).clone(),
                        mutable: reference.mutability.is_some(),
                    },
                    _ => return None,
                }
            } else if let Some(elem) = array_elem(&param.ty) {
                match classify(elem)? {
                    HarnessValue::Scalar(_) | HarnessValue::Generic(_) => {
                        HarnessValue::Array(elem.clone())
                    }
                    _ => return None,
                }
            } else {
                classify(&param.ty)?
            };
            params.push((param.name.clone(), value));
        }

        fn returnable(ty: &Type, classify: &dyn Fn(&Type) -> Option<HarnessValue>) -> bool {
            match ty {
                Type::Tuple(tuple) => {
                    !tuple.elems.is_empty() && tuple.elems.iter().all(|ty| returnable(ty, classify))
                }
                ty => match array_elem(ty) {
                    Some(elem) => matches!(
                        classify(elem),
                        Some(HarnessValue::Scalar(_) | HarnessValue::Generic(_))
                    ),
                    None => classify(ty).is_some(),
                },
            }
        }

        let KernelReturns::ExpandType(returns) = &sig.returns else {
            return None;
        };
        // A function that returns nothing is only worth a harness when it
        // writes into a slice the host gets back.
        let writes_back = params
            .iter()
            .any(|(_, value)| matches!(value, HarnessValue::Slice { mutable: true, .. }));
        let returns_unit = matches!(returns, Type::Tuple(tuple) if tuple.elems.is_empty());
        if !(returns_unit && writes_back || returnable(returns, &classify)) {
            return None;
        }

        Some(HarnessPlan {
            generics,
            params,
            returns: returns.clone(),
        })
    }

    fn harness_tokens(&self, plan: &HarnessPlan, gate: &HarnessGate) -> syn::Result<TokenStream> {
        let input_scalar = prelude_type("InputScalar");
        let buffer_arg = prelude_type("BufferArg");
        let array_type = prelude_type("Array");
        let cube_element = prelude_type("CubeElement");
        let launch_arg = prelude_type("LaunchArg");
        let harness_output = prelude_type("HarnessOutput");
        let store_output = prelude_type("__expand_store_output");
        let output_lens = prelude_type("__output_lens");
        let cube_kernel = prelude_type("CubeKernel");
        let cube_count = prelude_type("CubeCount");
        let cube_dim = prelude_type("CubeDim");
        let runtime = core_type("__private");
        let runtime = quote![#runtime::Runtime];
        let core = core_path();

        let name = &self.func.sig.name;
        let kernel_name = format_ident!("{name}_harness");
        let generic_names = plan.generics.iter().map(|(ident, _)| ident);
        let turbofish = match plan.generics.is_empty() {
            true => quote![],
            false => quote![::<#(#generic_names),*>],
        };
        // A generic must be a `CubeElement` to be read back, and a
        // `LaunchArg` for a cube type over it to be launched.
        let generics = plan
            .generics
            .iter()
            .map(|(ident, bound)| quote![#ident: #bound + #cube_element + #launch_arg])
            .collect::<Vec<_>>();
        let ret = &plan.returns;

        // The kernel calls the function through a path the cube parser turns
        // into a call to this module's `expand`. `super::expand` can't be used
        // directly, since the parser would append another `::expand`.
        let array_len = |param: &Ident| format_ident!("__harness_{param}_len");
        let mut kernel_params = Vec::new();
        let mut kernel_stmts = Vec::new();
        let mut call_args = Vec::new();
        for (param, value) in plan.params.iter() {
            match value {
                HarnessValue::Scalar(ty) | HarnessValue::Cube(ty) => {
                    kernel_params.push(quote![#param: #ty]);
                    call_args.push(quote![#param]);
                }
                HarnessValue::Slice { elem, mutable } => {
                    let mutability = mutable.then(|| quote![mut]);
                    kernel_params.push(quote![#param: &#mutability [#elem]]);
                    call_args.push(quote![#param]);
                }
                HarnessValue::Array(elem) => {
                    let len = array_len(param);
                    let array = format_ident!("__harness_{param}_array");
                    kernel_params.push(quote![#param: &[#elem]]);
                    kernel_params.push(quote![#[comptime] #len: usize]);
                    kernel_stmts.push(quote! {
                        let mut #array = #array_type::<#elem>::new(#len);
                        for __i in 0..#len {
                            #array[__i] = #param[__i];
                        }
                    });
                    call_args.push(quote![#array]);
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
            fn #kernel_name<#(#generics),*>(
                #(#kernel_params,)*
                mut __harness_out: <#ret as #harness_output>::Out,
            ) {
                #(#kernel_stmts)*
                let __harness_ret = __harness_call #turbofish(#(#call_args),*);
                intrinsic!(|scope| #store_output::<#ret>(
                    scope,
                    &mut __harness_out,
                    __harness_ret,
                ));
            }
        };
        let probe = returns_array(ret);
        let kernel_args = KernelArgs {
            launch: Flag::present(),
            create_dummy_kernel: match probe {
                true => Flag::present(),
                false => Flag::default(),
            },
            debug_symbols: self.args.debug_symbols,
            no_debug_symbols: self.args.no_debug_symbols,
            src_file: self.args.src_file.clone(),
            ..Default::default()
        };
        let kernel = Launch::from_item_fn(kernel, kernel_args)?;

        let harness_params = plan.params.iter().map(|(param, value)| match value {
            HarnessValue::Scalar(ty) | HarnessValue::Comptime(ty) | HarnessValue::Cube(ty) => {
                quote![#param: #ty]
            }
            HarnessValue::Generic(ident) => quote![#param: #ident],
            HarnessValue::Array(elem) => quote![#param: &[#elem]],
            HarnessValue::Slice { elem, mutable } => {
                let mutability = mutable.then(|| quote![mut]);
                quote![#param: &#mutability [#elem]]
            }
        });
        let slice_handle = |param: &Ident| format_ident!("__harness_{param}");
        let slices = plan
            .params
            .iter()
            .filter_map(|(param, value)| match value {
                HarnessValue::Slice { elem, mutable } => Some((param, elem, *mutable)),
                HarnessValue::Array(elem) => Some((param, elem, false)),
                _ => None,
            })
            .collect::<Vec<_>>();
        let uploads = slices.iter().map(|(param, elem, _)| {
            let handle = slice_handle(param);
            quote! {
                let #handle = __client.create_from_slice(<#elem as #cube_element>::as_bytes(#param));
            }
        });
        // A mutable slice is read back after the returned value, in the
        // order of the parameters.
        let write_backs = slices
            .iter()
            .filter(|(_, _, mutable)| *mutable)
            .collect::<Vec<_>>();
        let write_back_handles = write_backs.iter().map(|(param, _, _)| slice_handle(param));
        let write_back = write_backs.iter().map(|(param, elem, _)| {
            quote! {
                let __written = __bytes.next().expect("a harness buffer for every mutable slice");
                #param.copy_from_slice(
                    &<#elem as #cube_element>::from_bytes(&__written)[..#param.len()],
                );
            }
        });
        // The arguments are built twice when the kernel is expanded first, so
        // the probe takes a clone of every handle and cube type.
        let launch_args = |probe: bool| {
            let clone = probe.then(|| quote![.clone()]);
            plan.params
                .iter()
                .map(|(param, value)| match value {
                    HarnessValue::Generic(ident) => {
                        quote![#input_scalar::new(#param, <#ident as #cube_element>::cube_type())]
                    }
                    HarnessValue::Cube(ty) => quote! {
                        ::core::convert::Into::<<#ty as #launch_arg>::RuntimeArg>::into(#param #clone)
                    },
                    HarnessValue::Slice { .. } => {
                        let handle = slice_handle(param);
                        // SAFETY: the buffer was created from the slice, so it
                        // holds exactly its length.
                        quote![unsafe { #buffer_arg::from_raw_parts(#handle #clone, #param.len()) }]
                    }
                    HarnessValue::Array(_) => {
                        let handle = slice_handle(param);
                        // SAFETY: as for a slice. The length is also the array's.
                        quote! {
                            unsafe { #buffer_arg::from_raw_parts(#handle #clone, #param.len()) },
                            #param.len()
                        }
                    }
                    HarnessValue::Scalar(_) | HarnessValue::Comptime(_) => quote![#param],
                })
                .collect::<Vec<_>>()
        };
        let cube_bounds = plan.params.iter().filter_map(|(_, value)| match value {
            HarnessValue::Cube(ty) => {
                let clone = probe.then(|| quote![#ty: ::core::clone::Clone,]);
                Some(quote! {
                    #clone
                    <#ty as #launch_arg>::RuntimeArg: ::core::convert::From<#ty>
                })
            }
            _ => None,
        });
        // The probe expands the kernel with a placeholder output, the same
        // way a launch would, to learn the returned arrays' lengths.
        let lens = match probe {
            true => {
                let probe_args = launch_args(true);
                quote! {
                    let __lens = {
                        let __out = <#ret as #harness_output>::alloc(
                            &__client,
                            &mut ::core::iter::repeat(1),
                            &mut #core::__private::vec::Vec::new(),
                        );
                        let __kernel = #kernel_name::create_dummy_kernel #turbofish(
                            __client.properties_shared(),
                            __client.target_properties_shared(),
                            #cube_count::new_single(),
                            #cube_dim::new_single(),
                            #(#probe_args,)*
                            __out,
                        );
                        #output_lens(&#cube_kernel::define(&__kernel))
                    };
                }
            }
            false => quote![let __lens = #core::__private::vec::Vec::<usize>::new();],
        };
        let launch_args = launch_args(false);
        let name_str = name.to_string();
        let doc = format!(
            "Run [{name_str}()] on a single unit of the given runtime and return its result, \
             writing any mutable slice back.\n\n\
             # Panics\n\n\
             If the kernel fails to compile or run."
        );

        let gate = gate.attr();

        Ok(quote! {
            #gate
            #[doc(hidden)]
            pub mod __harness_call {
                pub use super::expand;
            }

            #gate
            #kernel

            #gate
            #[allow(dead_code, clippy::too_many_arguments, clippy::let_unit_value)]
            #[doc = #doc]
            pub fn harness<__R: #runtime, #(#generics),*>(
                #(#harness_params),*
            ) -> <#ret as #harness_output>::Host
            where
                #(#cube_bounds,)*
            {
                let __client = <__R as #runtime>::client(&::core::default::Default::default());
                #(#uploads)*
                #lens
                let mut __handles = #core::__private::vec::Vec::new();
                let __out = <#ret as #harness_output>::alloc(
                    &__client,
                    &mut __lens.iter().copied(),
                    &mut __handles,
                );
                #(__handles.push(#write_back_handles.clone());)*
                #kernel_name::launch #turbofish(
                    &__client,
                    #cube_count::new_single(),
                    #cube_dim::new_single(),
                    #(#launch_args,)*
                    __out,
                );
                let mut __bytes = __client
                    .read(__handles)
                    .unwrap_or_else(|err| panic!("harness for `{}` failed: {err}", #name_str))
                    .into_iter();
                let __ret = <#ret as #harness_output>::read(&mut __bytes, &mut __lens.iter().copied());
                #(#write_back)*
                __ret
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
        assert_eq!(plan.returns, parse_quote!(f32));
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
        assert_eq!(plan.returns, parse_quote!(F));
        assert_eq!(plan.params[1].1, HarnessValue::Generic(parse_quote!(I)));
    }

    #[test]
    fn where_bounds_are_inline_bounds() {
        let plan = plan(parse_quote! {
            fn f<F, I>(x: F, y: I) -> (F, I) where F: Float, I: Int { (x, y) }
        })
        .unwrap();

        assert_eq!(
            plan.generics,
            vec![
                (parse_quote!(F), parse_quote!(Float)),
                (parse_quote!(I), parse_quote!(Int)),
            ]
        );
        assert_eq!(plan.params[0].1, HarnessValue::Generic(parse_quote!(F)));
    }

    #[test]
    fn tuple_returns_are_pure() {
        let returns = |item: ItemFn| plan(item).unwrap().returns;

        assert_eq!(
            returns(parse_quote! { fn f<F: Float>(x: F) -> (F, u32, i64) { (x, 0, 0) } }),
            parse_quote!((F, u32, i64))
        );
        assert_eq!(
            returns(parse_quote! { fn f() -> (u32,) { (0,) } }),
            parse_quote!((u32,))
        );
        assert_eq!(
            returns(parse_quote! { fn f(x: u32) -> (u32, (u32, u32)) { (x, (x, x)) } }),
            parse_quote!((u32, (u32, u32)))
        );
    }

    #[test]
    fn cube_types_are_harnessable() {
        let plan = plan(parse_quote! {
            fn f<F: Float>(p: Point<F>, x: F) -> (Point<F>, u32) { (p, 0) }
        })
        .unwrap();
        assert_eq!(plan.params[0].1, HarnessValue::Cube(parse_quote!(Point<F>)));
        assert_eq!(plan.params[1].1, HarnessValue::Generic(parse_quote!(F)));
        assert_eq!(plan.returns, parse_quote!((Point<F>, u32)));
    }

    #[test]
    fn slices_are_harnessable() {
        let plan = plan(parse_quote! {
            fn f<F: Float>(xs: &[F], ys: &'a [u32], out: &mut [i64]) -> F { xs[0] }
        })
        .unwrap();

        assert_eq!(
            plan.params
                .iter()
                .map(|(_, v)| v.clone())
                .collect::<Vec<_>>(),
            vec![
                HarnessValue::Slice {
                    elem: parse_quote!(F),
                    mutable: false
                },
                HarnessValue::Slice {
                    elem: parse_quote!(u32),
                    mutable: false
                },
                HarnessValue::Slice {
                    elem: parse_quote!(i64),
                    mutable: true
                },
            ]
        );
    }

    #[test]
    fn arrays_are_harnessable() {
        let plan = plan(parse_quote! {
            fn f<F: Float>(xs: Array<F>, ys: cubecl::prelude::Array<u32>) -> F { xs[0] }
        })
        .unwrap();

        assert_eq!(plan.params[0].1, HarnessValue::Array(parse_quote!(F)));
        assert_eq!(plan.params[1].1, HarnessValue::Array(parse_quote!(u32)));
    }

    #[test]
    fn arrays_are_returnable() {
        let plan = plan(parse_quote! {
            fn f<F: Float>(xs: Array<F>) -> (Array<F>, u32, (Array<u8>,)) { (xs, 0, (xs,)) }
        })
        .unwrap();

        assert_eq!(plan.returns, parse_quote!((Array<F>, u32, (Array<u8>,))));
        assert!(returns_array(&plan.returns));
        assert!(!returns_array(&parse_quote!((u32, (f32,)))));
    }

    #[test]
    fn writing_into_a_slice_is_a_result() {
        let fill = plan(parse_quote! { fn f(x: u32, out: &mut [u32]) {} }).unwrap();
        assert_eq!(fill.returns, parse_quote!(()));

        let explicit = plan(parse_quote! { fn f(out: &mut [f32]) -> () {} }).unwrap();
        assert_eq!(explicit.returns, parse_quote!(()));
    }

    #[test]
    fn impure_functions_have_no_harness() {
        let impure: Vec<ItemFn> = vec![
            parse_quote! { fn f(x: u32) {} },
            parse_quote! { fn f(x: bool) -> u32 { 0 } },
            parse_quote! { fn f(x: u32) -> bool { true } },
            parse_quote! { fn f(x: u32) -> (u32, bool) { (x, true) } },
            parse_quote! { fn f(x: u32) -> (u32, ()) { (x, ()) } },
            parse_quote! { fn f(x: usize) -> u32 { 0 } },
            parse_quote! { fn f(x: &u32) -> u32 { 0 } },
            parse_quote! { fn f(x: &[u32]) {} },
            parse_quote! { fn f(x: &[bool]) -> u32 { 0 } },
            parse_quote! { fn f(x: &[usize]) -> u32 { 0 } },
            parse_quote! { fn f(x: &[Point]) -> u32 { 0 } },
            parse_quote! { fn f(x: &[[u32; 2]]) -> u32 { 0 } },
            parse_quote! { fn f(p: &Point) -> u32 { 0 } },
            parse_quote! { fn f(x: &mut u32) {} },
            parse_quote! { fn f(x: &mut [u32]) -> (u32, ()) { (0, ()) } },
            parse_quote! { fn f(x: Array<bool>) -> u32 { 0 } },
            parse_quote! { fn f(x: Array<Point>) -> u32 { 0 } },
            parse_quote! { fn f(x: &Array<u32>) -> u32 { 0 } },
            parse_quote! { fn f(x: u32) -> Array<bool> { 0 } },
            parse_quote! { fn f(x: u32) -> Array<Point> { 0 } },
            parse_quote! { fn f(x: u32) -> (u32, Array<usize>) { 0 } },
            parse_quote! { fn f(p: Point) {} },
            parse_quote! { fn f<N: Size>(x: u32) -> u32 { 0 } },
            parse_quote! { fn f<F: Float + CubeElement>(x: F) -> F { x } },
            parse_quote! { fn f<F: Float>(x: F) -> F where F: Copy { x } },
            parse_quote! { fn f<F>(x: F) -> F where F: Float + CubeElement { x } },
            parse_quote! { fn f<F>(x: F) -> F where F: Float, F: Copy { x } },
            parse_quote! { fn f<F: Float>(x: F) -> F where F::Expand: Clone { x } },
            parse_quote! { fn f<F: Float>(x: F) -> F where u32: Copy { x } },
            parse_quote! { fn f<F>(x: F) -> F where for<'a> F: Float { x } },
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
            fn f<F: Float, I: Int>(x: F, y: I, z: u32, #[comptime] c: u32) -> (F, I) { (x, y) }
        };
        let expand = || {
            let launch = Launch::from_item_fn(item.clone(), KernelArgs::default()).unwrap();
            let plan = launch.harness_plan().unwrap();
            launch
                .harness_tokens(&plan, &HarnessGate::Always)
                .unwrap()
                .to_string()
        };

        let first = expand();
        for _ in 0..16 {
            assert_eq!(expand(), first);
        }
    }
}
