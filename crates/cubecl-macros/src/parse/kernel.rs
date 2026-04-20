use crate::{
    RemoveHelpers,
    expression::{Block, Expression},
    parse::helpers::is_define_attribute,
    paths::{frontend_type, prelude_type},
    scope::Context,
    statement::{DefineKind, Pattern, Statement},
};
use core::hash::Hash;
use darling::{FromMeta, ast::NestedMeta, util::Flag};
use inflections::case::to_snake_case;
use proc_macro2::{Span, TokenStream};
use quote::{ToTokens, format_ident, quote};
use std::{collections::HashMap, iter};
use syn::{
    AssocType, ConstParam, Expr, FnArg, GenericArgument, Generics, Ident, ItemFn, LitStr, Path,
    ReturnType, Signature, TraitItemFn, Type, TypeGroup, TypeMacro, TypeParam, TypeParen,
    Visibility, parse, parse_quote, punctuated::Punctuated, spanned::Spanned, token::Mut,
    visit_mut::VisitMut,
};

use super::{desugar::Desugar, helpers::is_comptime_attr, statement::parse_pat};

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
                _ => todo!(),
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

#[derive(Clone, Debug)]
pub struct KernelSignature {
    pub name: Ident,
    pub parameters: Vec<KernelParam>,
    pub returns: KernelReturns,
    pub generics: Generics,
    pub receiver_arg: Option<FnArg>,
}

impl KernelSignature {
    pub fn runtime_params(&self) -> impl Iterator<Item = &KernelParam> {
        self.parameters.iter().filter(|it| !it.is_const)
    }

    pub fn define_mappings(&self) -> HashMap<Ident, (Ident, Option<usize>)> {
        let mut mapping = HashMap::new();
        for param in self.parameters.iter() {
            for define in param.defines.iter() {
                match define {
                    DefinedGeneric::Single(ident) => {
                        mapping.insert(ident.clone(), (param.name.clone(), None));
                    }
                    DefinedGeneric::Multiple(ident, index) => {
                        mapping.insert(ident.clone(), (param.name.clone(), Some(*index)));
                    }
                }
            }
        }
        mapping
    }
}

#[derive(Clone, Debug)]
pub enum KernelReturns {
    ExpandType(Type),
    Plain(Type),
}

impl KernelReturns {
    pub fn ty(&self) -> Type {
        match self {
            KernelReturns::ExpandType(ty) => ty.clone(),
            KernelReturns::Plain(ty) => ty.clone(),
        }
    }
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

#[derive(Clone, Debug)]
pub struct KernelParam {
    pub name: Ident,
    pub ty: Type,
    pub normalized_ty: Type,
    pub defines: Vec<DefinedGeneric>,
    pub is_const: bool,
    pub mutability: Option<Mut>,
}

impl KernelParam {
    pub fn from_param(param: FnArg) -> syn::Result<Self> {
        let param = match param {
            FnArg::Typed(param) => param,
            FnArg::Receiver(param) => {
                let normalized_ty = expand_kernel_ty(*param.ty.clone(), false);

                let mutability = if param.reference.is_none() {
                    param.mutability
                } else {
                    None
                };

                return Ok(KernelParam {
                    name: Ident::new("self", param.span()),
                    ty: *param.ty,
                    normalized_ty,
                    defines: Vec::new(),
                    is_const: false,
                    mutability,
                });
            }
        };
        let Pattern {
            ident,
            is_ref,
            mutability,
            ..
        } = parse_pat(*param.pat.clone())?;
        let mut is_const = false;
        let mut defines = Vec::new();

        for attr in param.attrs.iter() {
            if is_comptime_attr(attr) {
                is_const = true;
            }
            if is_define_attribute(attr) {
                match attr.parse_args::<Ident>() {
                    Ok(ident) => {
                        defines.push(DefinedGeneric::Single(ident));
                    }
                    Err(_) => {
                        let list = attr.meta.require_list().expect("Wrong syntax.");
                        let tokens = list.tokens.to_string();
                        let names = tokens.split(",");
                        for (i, name) in names.enumerate() {
                            let ident = Ident::new(name.trim(), attr.span());
                            defines.push(DefinedGeneric::Multiple(ident, i));
                        }
                    }
                };
                is_const = true;
            }
        }

        let ty = *param.ty.clone();
        let normalized_ty = expand_kernel_ty(*param.ty, is_const);

        let mut_token = if !is_ref { mutability } else { None };

        Ok(Self {
            name: ident,
            ty,
            defines,
            normalized_ty,
            is_const,
            mutability: mut_token,
        })
    }

    /// If the type is self, we set the normalized ty to self as well.
    ///
    /// Useful when the param is used in functions or methods associated to the
    /// expand type.
    pub fn plain_normalized_self(&mut self) {
        fn is_self(ty: &Type) -> bool {
            match ty {
                Type::Path(type_path) if type_path.path.is_ident("Self") => true,
                Type::Ptr(type_ptr) => is_self(&type_ptr.elem),
                Type::Reference(type_reference) => is_self(&type_reference.elem),
                _ => false,
            }
        }
        if is_self(&self.ty) {
            self.normalized_ty = self.ty.clone();
        }
    }
}

impl KernelSignature {
    pub fn from_signature(sig: Signature, args: &KernelArgs) -> syn::Result<Self> {
        let name = sig.ident;
        let mut generics = sig.generics;
        let returns = match sig.output {
            syn::ReturnType::Default => KernelReturns::ExpandType(parse_quote![()]),
            syn::ReturnType::Type(_, ty) => match *ty.clone() {
                Type::Macro(TypeMacro { mac }) => {
                    if mac.path.is_ident("comptime_type") {
                        let inner_type = parse::<Type>(mac.tokens.into())
                            .expect("Interior of comptime_type macro should be a valid type.");
                        KernelReturns::Plain(inner_type)
                    } else {
                        panic!("Only comptime_type macro supported on return type")
                    }
                }
                _ => KernelReturns::ExpandType(*ty),
            },
        };
        let receiver_arg = sig
            .inputs
            .iter()
            .find(|it| matches!(it, FnArg::Receiver(_)))
            .cloned();
        let sig_params = sig
            .inputs
            .into_iter()
            .map(KernelParam::from_param)
            .collect::<Result<Vec<_>, _>>()?;
        let manually_defined_params = sig_params
            .iter()
            .flat_map(|it| it.defines.iter().map(|it| it.ident()))
            .collect::<Vec<_>>();
        let define_params = generics
            .type_params()
            .filter(|it| !manually_defined_params.contains(&&it.ident))
            .filter(|it| {
                it.attrs.iter().any(is_define_attribute)
                    // define sizes by default on launch functions
                    || (args.is_launch() && it.bounds.to_token_stream().to_string() == "Size")
            })
            .map(|ty_param| {
                let type_ = prelude_type("Type");
                let ident = &ty_param.ident;
                let name = format_ident!("_{}", to_snake_case(&ident.to_string()));
                let is_size = ty_param.bounds.to_token_stream().to_string() == "Size";
                let ty = match is_size {
                    true => quote![usize],
                    false => quote![#type_],
                };
                KernelParam::from_param(parse_quote!(#[define(#ident)] #name: #ty))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let parameters = define_params
            .into_iter()
            .chain(sig_params)
            .collect::<Vec<_>>();

        RemoveHelpers.visit_generics_mut(&mut generics);

        Ok(KernelSignature {
            generics,
            name,
            parameters,
            returns,
            receiver_arg,
        })
    }

    pub fn from_trait_fn(function: TraitItemFn, args: &KernelArgs) -> syn::Result<Self> {
        Self::from_signature(function.sig, args)
    }

    /// If the type is self, we set the returns type to plain instead of expand
    /// type.
    pub fn plain_self(&mut self) {
        if let Type::Path(pat) = self.returns.ty()
            && pat.path.is_ident("Self")
        {
            self.returns = KernelReturns::Plain(self.returns.ty());
        }

        for param in self.parameters.iter_mut() {
            if let Type::Path(pat) = &param.ty
                && pat.path.is_ident("Self")
            {
                param.normalized_ty = parse_quote!(Self);
            }
        }
    }
}

impl KernelFn {
    pub fn from_sig_and_block(
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
