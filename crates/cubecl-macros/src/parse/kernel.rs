use crate::{
    expression::{Block, Expression},
    parse::signature::KernelSignature,
    paths::{frontend_type, prelude_type},
    scope::Context,
    statement::{DefineKind, Statement},
};
use core::hash::Hash;
use darling::{FromMeta, ast::NestedMeta, util::Flag};
use proc_macro2::{Span, TokenStream};
use quote::{ToTokens, format_ident, quote};
use std::{collections::HashMap, iter};
use syn::{
    AssocType, Attribute, ConstParam, Expr, GenericArgument, Generics, Ident, ItemFn, LitStr, Path,
    ReturnType, Signature, Type, TypeGroup, TypeParam, TypeParen, Visibility, parse_quote,
    punctuated::Punctuated, visit_mut::VisitMut,
};

use super::desugar::Desugar;

#[derive(Default, FromMeta, Clone)]
pub(crate) struct KernelArgs {
    pub launch: Flag,
    pub launch_unchecked: Flag,
    pub debug_symbols: Flag,
    // Force override because macro hygiene can cause weird issues
    pub no_debug_symbols: Flag,
    pub fast_math: Option<Expr>,
    pub debug: Flag,
    pub create_dummy_kernel: Flag,
    /// Generate expansion only, for expanding existing types
    pub expand_only: Flag,
    pub cluster_dim: Option<Expr>,
    pub src_file: Option<LitStr>,
    /// Base traits for a split expand trait
    pub expand_base_traits: Option<String>,
    /// Pass define types explicitly, to allow concrete types instead of auto-generating them
    pub explicit_define: Flag,
    #[darling(default)]
    pub address_type: AddressType,
}

#[derive(Default, FromMeta, PartialEq, Eq, Clone, Copy)]
pub(crate) enum AddressType {
    #[default]
    U32,
    U64,
    Dynamic,
}

pub fn from_tokens<T: FromMeta>(tokens: TokenStream) -> syn::Result<T> {
    let meta = NestedMeta::parse_meta_list(tokens)?;
    T::from_list(&meta).map_err(syn::Error::from)
}

impl KernelArgs {
    pub fn is_launch(&self) -> bool {
        self.launch.is_present() || self.launch_unchecked.is_present()
    }
}

#[derive(Clone)]
pub struct GenericArg {
    pub expand_ty: syn::Path,
    pub marker_ty: syn::Ident,
    pub kind: DefineKind,
}

#[derive(Clone)]
pub struct GenericAnalysis {
    pub map: HashMap<syn::Ident, GenericArg>,
}

impl GenericAnalysis {
    pub fn process_generic_names(&self, ty: &syn::Generics) -> TokenStream {
        let mut output = quote![];

        if ty.params.is_empty() {
            return output;
        }

        for param in ty.params.iter() {
            match param {
                syn::GenericParam::Type(TypeParam { ident, .. })
                | syn::GenericParam::Const(ConstParam { ident, .. }) => {
                    if let Some(GenericArg { expand_ty, .. }) = self.map.get(ident) {
                        output.extend(quote![#expand_ty,]);
                    } else {
                        output.extend(quote![#ident,]);
                    }
                }
                other => output.extend(quote![#other,]),
            }
        }

        quote! {
            ::<#output>
        }
    }

    pub fn register_types(
        &self,
        mut name_mapping: HashMap<Ident, (Ident, Option<usize>)>,
        scope: TokenStream,
        has_self: bool,
        launch: bool,
    ) -> TokenStream {
        let mut output = quote![];
        let self_ = has_self.then(|| quote![self.]);

        for (
            ident,
            GenericArg {
                kind, expand_ty, ..
            },
        ) in self.map.iter()
        {
            let name = match name_mapping.remove(ident) {
                Some((name, index)) => match index {
                    Some(index) => {
                        // The defined type should be an array or vector that support indexing.
                        quote! { #self_ #name[#index].into() }
                    }
                    None => quote! { #self_ #name.into() },
                },
                None if !launch => {
                    continue;
                }
                None => match kind {
                    DefineKind::Type => quote![#ident::as_type_native_unchecked().storage_type()],
                    DefineKind::Size => quote![#ident::value()],
                },
            };
            match kind {
                DefineKind::Type => {
                    output.extend(quote! {
                        #scope.register_type::<#expand_ty>(#name);
                    });
                }
                DefineKind::Size => {
                    output.extend(quote! {
                        #scope.register_size::<#expand_ty>(#name);
                    });
                }
            }
        }
        if !name_mapping.is_empty() {
            for key in name_mapping.keys() {
                let err = syn::Error::new_spanned(
                    key,
                    format!("Generic `{key}` isn't defined correctly. Only `Float`, `Int` and `Numeric` generics can be defined with only a single trait bound."),
                ).into_compile_error();
                output.extend(err);
            }
        }

        output
    }

    pub fn process_ty(&self, ty: &syn::Type) -> syn::Type {
        fn process_ty_inner(ty: &mut Type, map: &HashMap<syn::Ident, GenericArg>) {
            match ty {
                Type::Array(type_array) => process_ty_inner(&mut type_array.elem, map),
                Type::Group(type_group) => process_ty_inner(&mut type_group.elem, map),
                Type::Paren(type_paren) => process_ty_inner(&mut type_paren.elem, map),
                Type::Path(type_path) => {
                    if let Some(GenericArg { expand_ty, .. }) =
                        type_path.path.get_ident().and_then(|ident| map.get(ident))
                    {
                        type_path.path = expand_ty.clone();
                    }
                    let generics = all_path_args_mut(&mut type_path.path);
                    for generic in generics {
                        process_generic_param_inner(generic, map);
                    }
                }
                Type::Ptr(type_ptr) => process_ty_inner(&mut type_ptr.elem, map),
                Type::Reference(type_reference) => process_ty_inner(&mut type_reference.elem, map),
                Type::Slice(type_slice) => process_ty_inner(&mut type_slice.elem, map),
                Type::Tuple(type_tuple) => {
                    for ty in type_tuple.elems.iter_mut() {
                        process_ty_inner(ty, map);
                    }
                }
                // Maybe implement this later
                Type::ImplTrait(_) => {}
                _ => {}
            }
        }

        fn process_generic_param_inner(
            arg: &mut GenericArgument,
            map: &HashMap<syn::Ident, GenericArg>,
        ) {
            match arg {
                GenericArgument::Type(Type::Path(path))
                    if path
                        .path
                        .get_ident()
                        .is_some_and(|ident| map.contains_key(ident)) =>
                {
                    let GenericArg { expand_ty, .. } =
                        map.get(path.path.get_ident().unwrap()).unwrap();
                    path.path = expand_ty.clone();
                }
                GenericArgument::Type(ty) => process_ty_inner(ty, map),
                GenericArgument::AssocType(AssocType {
                    generics: Some(generics),
                    ..
                }) => {
                    for arg in generics.args.iter_mut() {
                        process_generic_param_inner(arg, map);
                    }
                }
                _ => {}
            }
        }
        let mut ty = ty.clone();
        process_ty_inner(&mut ty, &self.map);
        ty
    }

    pub fn from_generics(generics: &syn::Generics, explicit_defines: bool) -> Self {
        let mut map = HashMap::new();
        let elem_expand = prelude_type("DynamicScalar");
        let size_expand = prelude_type("DynamicSize");

        for type_param in generics.type_params() {
            if type_param.bounds.len() > 1 {
                continue;
            }

            if let Some(syn::TypeParamBound::Trait(trait_bound)) = type_param.bounds.first()
                && let Some(bound) = trait_bound.path.get_ident()
            {
                let name = bound.to_string();
                let ident = type_param.ident.clone();
                let marker_ty = format_ident!("_{ident}");

                match name.as_str() {
                    "Float" | "Int" | "Numeric" | "CubePrimitive" => {
                        if explicit_defines {
                            map.insert(
                                ident.clone(),
                                GenericArg {
                                    expand_ty: parse_quote!(#ident),
                                    marker_ty,
                                    kind: DefineKind::Type,
                                },
                            );
                        } else {
                            map.insert(
                                ident,
                                GenericArg {
                                    expand_ty: parse_quote!(#elem_expand<#marker_ty>),
                                    marker_ty,
                                    kind: DefineKind::Type,
                                },
                            );
                        }
                    }
                    "Size" => {
                        if explicit_defines {
                            map.insert(
                                ident.clone(),
                                GenericArg {
                                    expand_ty: parse_quote!(#ident),
                                    marker_ty,
                                    kind: DefineKind::Size,
                                },
                            );
                        } else {
                            map.insert(
                                type_param.ident.clone(),
                                GenericArg {
                                    expand_ty: parse_quote!(#size_expand<#marker_ty>),
                                    marker_ty,
                                    kind: DefineKind::Size,
                                },
                            );
                        }
                    }
                    _ => {}
                };
            };
        }

        Self { map }
    }
}

pub struct Launch {
    pub args: KernelArgs,
    pub vis: Visibility,
    pub func: KernelFn,
    pub kernel_generics: Generics,
    pub launch_generics: Generics,
}

#[derive(Clone)]
pub struct KernelFn {
    pub attrs: Vec<Attribute>,
    pub vis: Visibility,
    pub sig: KernelSignature,
    pub body: KernelBody,
    pub full_name: String,
    pub span: Span,
    pub context: Context,
    pub args: KernelArgs,
    pub analysis: GenericAnalysis,
}

#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
pub enum KernelBody {
    Block(Block),
    Verbatim(TokenStream),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// A generic type that is mapped to a runtime values that implements Into<StorageType>.
pub enum DefinedGeneric {
    /// The define annotation is on a single generic element.
    Single(Ident),
    /// The define annotation is on multiple generic elements.
    ///
    /// The runtime value that maps multiple generics must be an array.
    Multiple(Ident, usize),
}

impl DefinedGeneric {
    /// Whether the defined type contains the provided ident.
    pub fn contains_ident(&self, ident_input: &Ident) -> bool {
        self.ident() == ident_input
    }

    /// Retrieves the ident of the defined generic
    pub fn ident(&self) -> &Ident {
        match self {
            DefinedGeneric::Single(ident, ..) => ident,
            DefinedGeneric::Multiple(ident, ..) => ident,
        }
    }
}

impl KernelFn {
    pub fn from_sig_and_block(
        attrs: Vec<Attribute>,
        vis: Visibility,
        sig: Signature,
        mut block: syn::Block,
        full_name: String,
        args: &KernelArgs,
    ) -> syn::Result<Self> {
        let cfg_debug = cfg!(debug_symbols) && !args.no_debug_symbols.is_present();
        let debug_symbols = cfg_debug || args.debug_symbols.is_present();

        let span = Span::call_site();
        let sig = KernelSignature::from_signature(sig, args)?;

        let analysis =
            GenericAnalysis::from_generics(&sig.generics, args.explicit_define.is_present());

        let mut context = Context::new(sig.returns.ty(), debug_symbols);
        context.extend(sig.parameters.clone());

        Desugar.visit_block_mut(&mut block);

        let (mut block, _) = context.in_scope(|ctx| Block::from_block(block, ctx))?;

        Self::patch_mut_owned_inputs(&mut block, &sig);

        Ok(KernelFn {
            attrs,
            vis,
            sig,
            body: KernelBody::Block(block),
            full_name,
            span,
            context,
            analysis,
            args: args.clone(),
        })
    }

    /// We need to call `IntoMut::into_mut` on mutable owned inputs since their
    /// local variables need to be identified as mut, which is done at
    /// initialization.
    ///
    /// However, we don't specify mutability during initialization when we don't
    /// need to mutate the type in the current scope; it is done in the
    /// function that receives the mutable parameter as input. Therefore, we
    /// need to adjust the mutability here.
    fn patch_mut_owned_inputs(block: &mut Block, sig: &KernelSignature) {
        let mut mappings = Vec::new();
        let into_mut = frontend_type("IntoMut");

        for s in sig.parameters.iter() {
            if s.mutability.is_some() {
                let name = s.name.clone();
                let expression = Expression::Verbatim {
                    tokens: quote! {
                        let mut #name = #into_mut::into_mut(#name, scope);
                    },
                };
                let stmt = Statement::Expression {
                    expression: Box::new(expression),
                    terminated: false,
                };
                mappings.push(stmt);
            }
        }

        if !mappings.is_empty() {
            mappings.append(&mut block.inner);
            block.inner = mappings;
        }
    }
}

impl Launch {
    pub fn from_item_fn(function: ItemFn, args: KernelArgs) -> syn::Result<Self> {
        let runtime = prelude_type("Runtime");
        let ret = function.sig.output.clone();

        let vis = function.vis;
        let full_name = function.sig.ident.to_string();
        let mut func = KernelFn::from_sig_and_block(
            // When generating code, this function will be wrapped in
            // a module. By setting the visibility to pub here, we
            // ensure that the function is visible outside that
            // module.
            function.attrs,
            Visibility::Public(parse_quote![pub]),
            function.sig,
            *function.block,
            full_name,
            &args,
        )?;

        // Bail early if the user tries to have a return type in a launch kernel.
        if args.is_launch()
            && let ReturnType::Type(arrow, ty) = &ret
        {
            // Span both the arrow and the return type
            let mut ts = arrow.to_token_stream();
            ts.extend(ty.into_token_stream());

            return Err(syn::Error::new_spanned(
                ts,
                format!(
                    "This is a launch kernel and cannot have a return type. Remove `-> {}`. Use mutable output arguments instead in order to get values out from kernels.",
                    ty.into_token_stream()
                ),
            ));
        }

        let mut kernel_generics = func.sig.generics.clone();
        kernel_generics.params.clear();
        let explicit_define = args.explicit_define.is_present();

        for param in func.sig.generics.params.iter_mut() {
            // We remove generic arguments based on defined types.
            let is_defined = |ident| {
                func.sig
                    .parameters
                    .iter()
                    .any(|p| p.defines.iter().any(|d| d.contains_ident(ident)))
            };
            match param.clone() {
                syn::GenericParam::Type(TypeParam { ident, .. })
                    if is_defined(&ident) && !explicit_define => {}
                param => {
                    kernel_generics.params.push(param);
                }
            }
        }

        kernel_generics.params.push(parse_quote![__R: #runtime]);
        let mut launch_generics = kernel_generics.clone();
        launch_generics.params =
            Punctuated::from_iter(iter::once(parse_quote!['kernel]).chain(launch_generics.params));

        Ok(Launch {
            args,
            vis,
            func,
            launch_generics,
            kernel_generics,
        })
    }
}

pub fn expand_kernel_ty(mut ty: Type, is_const: bool) -> Type {
    if is_const {
        ty
    } else {
        let cube_type = prelude_type("CubeType");
        map_type_normalized(&mut ty, &|ty| parse_quote![<#ty as #cube_type>::ExpandType]);
        ty
    }
}

pub fn map_type_normalized(ty: &mut Type, op: &impl Fn(&Type) -> Type) {
    match ty {
        Type::Group(type_group) => map_type_normalized(&mut type_group.elem, op),
        Type::ImplTrait(_) => {
            unimplemented!("impl trait not yet supported in kernel args")
        }
        // Probably won't work with inference but we can at least try
        Type::Infer(_) => {}
        // Do nothing
        Type::Macro(type_macro) if type_macro.mac.path.is_ident("comptime_type") => {}
        Type::Macro(_) => {
            unimplemented!("Macro types not allowed for kernel args")
        }
        Type::Never(_) => {}
        Type::Paren(type_paren) => map_type_normalized(&mut type_paren.elem, op),
        Type::Ptr(type_ptr) => map_type_normalized(&mut type_ptr.elem, op),
        Type::Reference(type_reference) => map_type_normalized(&mut type_reference.elem, op),
        Type::TraitObject(_) => {
            unimplemented!("Trait objects are not allowed for kernel args")
        }
        // the `ExpandType` of tuple equals the expand type of each constituent, so we can
        // probably support more cases by handling each contained type separately. Won't hurt
        // at least.
        Type::Tuple(type_tuple) => {
            for ty in type_tuple.elems.iter_mut() {
                map_type_normalized(ty, op);
            }
        }
        Type::Verbatim(_) => {}
        other => {
            *other = op(other);
        }
    }
}

pub fn all_path_args_mut(path: &mut Path) -> impl Iterator<Item = &mut GenericArgument> {
    path.segments
        .iter_mut()
        .flat_map(|it| match &mut it.arguments {
            syn::PathArguments::AngleBracketed(args) => args.args.iter_mut().collect::<Vec<_>>(),
            _ => vec![],
        })
}

pub fn strip_ref(ty: Type) -> Type {
    match ty {
        Type::Reference(reference) => strip_ref(*reference.elem),
        Type::Ptr(ptr) => strip_ref(*ptr.elem),
        Type::Group(paren) => Type::Group(TypeGroup {
            elem: Box::new(strip_ref(*paren.elem)),
            ..paren
        }),
        Type::Paren(paren) => Type::Paren(TypeParen {
            elem: Box::new(strip_ref(*paren.elem)),
            ..paren
        }),
        ty => ty,
    }
}

pub fn anon_lifetime_to_static(mut ty: Type) -> Type {
    fn map_ty(ty: &mut Type) {
        match ty {
            Type::Array(type_array) => map_ty(&mut type_array.elem),
            Type::Group(type_group) => map_ty(&mut type_group.elem),
            Type::Paren(type_paren) => map_ty(&mut type_paren.elem),
            Type::Path(type_path) => map_path(&mut type_path.path),
            Type::Ptr(type_ptr) => map_ty(&mut type_ptr.elem),
            Type::Reference(reference)
                if reference.lifetime.as_ref().is_none_or(|l| l.ident == "_") =>
            {
                reference.lifetime = Some(parse_quote!('static));
            }
            Type::Slice(type_slice) => map_ty(&mut type_slice.elem),
            Type::Tuple(type_tuple) => {
                for ty in type_tuple.elems.iter_mut() {
                    map_ty(ty);
                }
            }
            _ => {}
        }
    }

    fn map_path(path: &mut Path) {
        for arg in path.segments.iter_mut().map(|it| &mut it.arguments) {
            if let syn::PathArguments::AngleBracketed(args) = arg {
                for arg in args.args.iter_mut() {
                    map_arg(arg)
                }
            }
        }
    }

    fn map_arg(arg: &mut GenericArgument) {
        match arg {
            GenericArgument::Lifetime(lifetime) if lifetime.ident == "_" => {
                *lifetime = parse_quote!('static);
            }
            GenericArgument::Type(ty) => map_ty(ty),
            _ => {}
        }
    }

    map_ty(&mut ty);
    ty
}
