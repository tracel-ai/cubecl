use std::collections::{HashMap, HashSet};

use inflections::case::to_snake_case;
use quote::{ToTokens, format_ident, quote};
use syn::{
    FnArg, GenericArgument, GenericParam, Generics, Ident, Lifetime, Path, Signature, TraitItemFn,
    Type, TypeMacro, TypeParamBound, parse, parse_quote, punctuated::Punctuated, spanned::Spanned,
    token::Mut, visit_mut::VisitMut,
};

use crate::{
    RemoveHelpers,
    parse::{
        helpers::{is_comptime_attr, is_define_attribute},
        kernel::{DefinedGeneric, KernelArgs, expand_kernel_ty},
        statement::parse_pat,
    },
    paths::prelude_type,
    statement::Pattern,
};

#[derive(Clone, Debug)]
pub struct KernelSignature {
    pub name: Ident,
    pub parameters: Vec<KernelParam>,
    pub returns: KernelReturns,
    pub generics: Generics,
    pub receiver_arg: Option<FnArg>,
    pub scope_lifetime: Option<Lifetime>,
}

impl KernelSignature {
    pub fn from_signature(mut sig: Signature, args: &KernelArgs) -> syn::Result<Self> {
        let name = sig.ident;
        let mut generics = sig.generics;
        let num_lifetime_params = generics.lifetimes().count();

        let input_lifetimes: HashSet<Lifetime> = sig
            .inputs
            .iter()
            .flat_map(|arg| match arg {
                FnArg::Receiver(receiver) => type_used_lifetimes(&receiver.ty),
                FnArg::Typed(pat_type) => type_used_lifetimes(&pat_type.ty),
            })
            .collect();
        let output_lifetimes = match &sig.output {
            syn::ReturnType::Default => HashSet::new(),
            syn::ReturnType::Type(_, ty) => type_used_lifetimes(ty),
        };

        // Need a separate lifetime for the scope if the output uses any lifetimes, or if we have
        // explicit input params (because mixing can cause issues with early vs late binding)
        let needs_explicit_lifetimes = !output_lifetimes.is_empty() || num_lifetime_params > 0;
        let has_inferred_lifetimes = input_lifetimes.contains(&parse_quote!('_))
            || output_lifetimes.contains(&parse_quote!('_));

        let mut scope_lifetime = None;

        if needs_explicit_lifetimes {
            let lifetime: Lifetime = parse_quote!('scope);
            generics
                .params
                .insert(num_lifetime_params, parse_quote!(#lifetime));
            scope_lifetime = Some(lifetime);
        }

        if needs_explicit_lifetimes && has_inferred_lifetimes {
            generics.params.insert(0, parse_quote!('infer));
            for input in sig.inputs.iter_mut() {
                match input {
                    FnArg::Receiver(receiver) => {
                        type_patch_inferred_lifetimes(&mut receiver.ty);
                        if let Some((token, lifetime)) = receiver.reference.as_mut() {
                            match lifetime {
                                Some(lifetime) if lifetime.ident == "_" => {
                                    lifetime.ident = Ident::new("infer", lifetime.ident.span());
                                }
                                Some(_) => {}
                                other => {
                                    *other = Some(Lifetime::new("'infer", token.span()));
                                }
                            }
                        }
                    }
                    FnArg::Typed(pat_type) => type_patch_inferred_lifetimes(&mut pat_type.ty),
                }
            }
            match &mut sig.output {
                syn::ReturnType::Default => {}
                syn::ReturnType::Type(_, ty) => type_patch_inferred_lifetimes(ty),
            }
        }

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
            scope_lifetime,
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

    pub fn call_generics(&self) -> Generics {
        let mut generics = self.generics.clone();
        generics.params = generics
            .params
            .into_iter()
            .filter(|param| !matches!(param, GenericParam::Lifetime(..)))
            .collect();
        generics
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

fn type_used_lifetimes(ty: &Type) -> HashSet<Lifetime> {
    match ty {
        Type::Array(type_array) => type_used_lifetimes(&type_array.elem),
        Type::Group(type_group) => type_used_lifetimes(&type_group.elem),
        Type::ImplTrait(type_impl_trait) => type_impl_trait
            .bounds
            .iter()
            .flat_map(type_param_bound_used_lifetimes)
            .collect(),
        Type::Paren(type_paren) => type_used_lifetimes(&type_paren.elem),
        Type::Path(type_path) => path_used_lifetimes(&type_path.path),
        Type::Ptr(type_ptr) => type_used_lifetimes(&type_ptr.elem),
        Type::Reference(type_reference) => {
            let lifetime = type_reference
                .lifetime
                .clone()
                .unwrap_or_else(|| Lifetime::new("'_", type_reference.and_token.span()));
            [lifetime].into_iter().collect()
        }
        Type::Slice(type_slice) => type_used_lifetimes(&type_slice.elem),
        Type::TraitObject(type_trait_object) => type_trait_object
            .bounds
            .iter()
            .flat_map(type_param_bound_used_lifetimes)
            .collect(),
        Type::Tuple(type_tuple) => type_tuple
            .elems
            .iter()
            .flat_map(type_used_lifetimes)
            .collect(),
        _ => HashSet::default(),
    }
}

fn type_param_bound_used_lifetimes(bound: &TypeParamBound) -> HashSet<Lifetime> {
    let mut out = HashSet::default();
    match bound {
        TypeParamBound::Trait(trait_bound) => {
            out.extend(path_used_lifetimes(&trait_bound.path));
        }
        TypeParamBound::Lifetime(lifetime) => {
            out.insert(lifetime.clone());
        }
        _ => {}
    }
    out
}

fn path_used_lifetimes(path: &Path) -> HashSet<Lifetime> {
    let mut out = HashSet::default();

    for arg in path.segments.iter().flat_map(|it| match &it.arguments {
        syn::PathArguments::AngleBracketed(args) => args.args.clone(),
        _ => Punctuated::new(),
    }) {
        out.extend(generic_arg_used_lifetimes(&arg));
    }

    out
}

fn generic_arg_used_lifetimes(arg: &GenericArgument) -> HashSet<Lifetime> {
    let mut out = HashSet::new();
    match arg {
        syn::GenericArgument::Lifetime(lifetime) => {
            out.insert(lifetime.clone());
        }
        syn::GenericArgument::Type(ty) => out.extend(type_used_lifetimes(ty)),
        syn::GenericArgument::AssocType(assoc_type) => {
            for arg in assoc_type.generics.iter().flat_map(|it| it.args.iter()) {
                out.extend(generic_arg_used_lifetimes(arg));
            }
            out.extend(type_used_lifetimes(&assoc_type.ty))
        }
        syn::GenericArgument::AssocConst(assoc_const) => {
            for arg in assoc_const.generics.iter().flat_map(|it| it.args.iter()) {
                out.extend(generic_arg_used_lifetimes(arg));
            }
        }
        syn::GenericArgument::Constraint(constraint) => {
            for arg in constraint.generics.iter().flat_map(|it| it.args.iter()) {
                out.extend(generic_arg_used_lifetimes(arg));
            }
            for bound in constraint.bounds.iter() {
                out.extend(type_param_bound_used_lifetimes(bound));
            }
        }
        _ => {}
    }
    out
}

fn type_patch_inferred_lifetimes(ty: &mut Type) {
    match ty {
        Type::Array(type_array) => type_patch_inferred_lifetimes(&mut type_array.elem),
        Type::Group(type_group) => type_patch_inferred_lifetimes(&mut type_group.elem),
        Type::ImplTrait(type_impl_trait) => {
            for bound in type_impl_trait.bounds.iter_mut() {
                type_param_bound_patch_inferred_lifetimes(bound);
            }
        }
        Type::Paren(type_paren) => type_patch_inferred_lifetimes(&mut type_paren.elem),
        Type::Path(type_path) => path_patch_inferred_lifetimes(&mut type_path.path),
        Type::Ptr(type_ptr) => type_patch_inferred_lifetimes(&mut type_ptr.elem),
        Type::Reference(type_reference) => {
            let span = type_reference.and_token.span();
            match &mut type_reference.lifetime {
                lifetime @ None => {
                    *lifetime = Some(Lifetime::new("'infer", span));
                }
                Some(lifetime) if lifetime.ident == "_" => {
                    lifetime.ident = Ident::new("infer", lifetime.ident.span());
                }
                _ => {}
            }
        }
        Type::Slice(type_slice) => type_patch_inferred_lifetimes(&mut type_slice.elem),
        Type::TraitObject(type_trait_object) => {
            for bound in type_trait_object.bounds.iter_mut() {
                type_param_bound_patch_inferred_lifetimes(bound);
            }
        }
        Type::Tuple(type_tuple) => {
            for ty in type_tuple.elems.iter_mut() {
                type_patch_inferred_lifetimes(ty);
            }
        }
        _ => {}
    }
}

fn type_param_bound_patch_inferred_lifetimes(bound: &mut TypeParamBound) {
    match bound {
        TypeParamBound::Trait(trait_bound) => {
            path_patch_inferred_lifetimes(&mut trait_bound.path);
        }
        TypeParamBound::Lifetime(lifetime) if lifetime.ident == "_" => {
            lifetime.ident = Ident::new("infer", lifetime.ident.span());
        }
        _ => {}
    }
}

fn path_patch_inferred_lifetimes(path: &mut Path) {
    for arg in path
        .segments
        .iter_mut()
        .filter_map(|it| match &mut it.arguments {
            syn::PathArguments::AngleBracketed(args) => Some(&mut args.args),
            _ => None,
        })
        .flatten()
    {
        generic_arg_patch_inferred_lifetimes(arg);
    }
}

fn generic_arg_patch_inferred_lifetimes(arg: &mut GenericArgument) {
    match arg {
        syn::GenericArgument::Lifetime(lifetime) if lifetime.ident == "_" => {
            lifetime.ident = Ident::new("infer", lifetime.ident.span());
        }
        syn::GenericArgument::Type(ty) => type_patch_inferred_lifetimes(ty),
        syn::GenericArgument::AssocType(assoc_type) => {
            for arg in assoc_type
                .generics
                .iter_mut()
                .flat_map(|it| it.args.iter_mut())
            {
                generic_arg_patch_inferred_lifetimes(arg);
            }
            type_patch_inferred_lifetimes(&mut assoc_type.ty);
        }
        syn::GenericArgument::AssocConst(assoc_const) => {
            for arg in assoc_const
                .generics
                .iter_mut()
                .flat_map(|it| it.args.iter_mut())
            {
                generic_arg_patch_inferred_lifetimes(arg);
            }
        }
        syn::GenericArgument::Constraint(constraint) => {
            for arg in constraint
                .generics
                .iter_mut()
                .flat_map(|it| it.args.iter_mut())
            {
                generic_arg_patch_inferred_lifetimes(arg);
            }
            for bound in constraint.bounds.iter_mut() {
                type_param_bound_patch_inferred_lifetimes(bound)
            }
        }
        _ => {}
    }
}
