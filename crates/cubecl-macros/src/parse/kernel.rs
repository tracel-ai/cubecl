use crate::{
    expression::Block,
    paths::prelude_type,
    scope::{Context, Scope},
    statement::{parse_pat, Pattern},
};
use darling::{ast::NestedMeta, util::Flag, FromMeta};
use proc_macro2::TokenStream;
use std::iter;
use syn::{
    parse_quote, punctuated::Punctuated, FnArg, Generics, Ident, ItemFn, Signature, TraitItemFn,
    Type, Visibility,
};

use super::helpers::is_comptime_attr;

#[derive(Default, FromMeta)]
pub(crate) struct KernelArgs {
    pub launch: Flag,
    pub launch_unchecked: Flag,
    pub debug: Flag,
    pub create_dummy_kernel: Flag,
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

pub struct Launch {
    pub args: KernelArgs,
    pub vis: Visibility,
    pub func: KernelFn,
    pub kernel_generics: Generics,
    pub launch_generics: Generics,
}

#[derive(Clone)]
pub struct KernelFn {
    pub sig: KernelSignature,
    pub block: Block,
    pub scope: Scope,
    pub context: Context,
}

#[derive(Clone)]
pub struct KernelSignature {
    pub name: Ident,
    pub parameters: Vec<KernelParam>,
    pub returns: Type,
    pub generics: Generics,
}

#[derive(Clone)]
pub struct KernelParam {
    pub name: Ident,
    pub ty: Type,
    pub normalized_ty: Type,
    pub is_const: bool,
    pub is_mut: bool,
    pub is_ref: bool,
}

impl KernelParam {
    fn from_param(param: FnArg) -> syn::Result<Self> {
        let param = match param {
            FnArg::Typed(param) => param,
            param => Err(syn::Error::new_spanned(
                param,
                "Can't use `cube` on methods",
            ))?,
        };
        let Pattern {
            ident,
            mut is_ref,
            mut is_mut,
            ..
        } = parse_pat(*param.pat.clone())?;
        let is_const = param.attrs.iter().any(is_comptime_attr);
        let ty = *param.ty.clone();
        let normalized_ty = normalize_kernel_ty(*param.ty, is_const, &mut is_ref, &mut is_mut);

        Ok(Self {
            name: ident,
            ty,
            normalized_ty,
            is_const,
            is_mut,
            is_ref,
        })
    }

    pub fn ty_owned(&self) -> Type {
        strip_ref(self.ty.clone(), &mut false, &mut false)
    }
}

impl KernelSignature {
    pub fn from_signature(sig: Signature) -> syn::Result<Self> {
        let name = sig.ident;
        let generics = sig.generics;
        let returns = match sig.output {
            syn::ReturnType::Default => parse_quote![()],
            syn::ReturnType::Type(_, ty) => *ty,
        };
        let parameters = sig
            .inputs
            .into_iter()
            .map(KernelParam::from_param)
            .collect::<Result<Vec<_>, _>>()?;

        Ok(KernelSignature {
            generics,
            name,
            parameters,
            returns,
        })
    }

    pub fn from_trait_fn(function: TraitItemFn) -> syn::Result<Self> {
        let name = function.sig.ident;
        let generics = function.sig.generics;
        let returns = match function.sig.output {
            syn::ReturnType::Default => parse_quote![()],
            syn::ReturnType::Type(_, ty) => *ty,
        };
        let parameters = function
            .sig
            .inputs
            .into_iter()
            .map(KernelParam::from_param)
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            generics,
            name,
            parameters,
            returns,
        })
    }
}

impl KernelFn {
    pub fn from_sig_and_block(sig: Signature, block: syn::Block) -> syn::Result<Self> {
        let sig = KernelSignature::from_signature(sig)?;

        let mut context = Context::new(sig.returns.clone());
        context.extend(sig.parameters.clone());
        let (block, scope) = context.with_scope(|ctx| Block::from_block(block, ctx));

        Ok(KernelFn {
            sig,
            block: block?,
            scope,
            context,
        })
    }
}

impl Launch {
    pub fn from_item_fn(function: ItemFn, args: KernelArgs) -> syn::Result<Self> {
        let runtime = prelude_type("Runtime");

        let vis = function.vis;
        let func = KernelFn::from_sig_and_block(function.sig, *function.block)?;
        let mut kernel_generics = func.sig.generics.clone();
        kernel_generics.params.push(parse_quote![__R: #runtime]);
        let mut expand_generics = kernel_generics.clone();
        expand_generics.params =
            Punctuated::from_iter(iter::once(parse_quote!['kernel]).chain(expand_generics.params));

        Ok(Launch {
            args,
            vis,
            func,
            kernel_generics,
            launch_generics: expand_generics,
        })
    }
}

fn normalize_kernel_ty(ty: Type, is_const: bool, is_ref: &mut bool, is_mut: &mut bool) -> Type {
    let ty = strip_ref(ty, is_ref, is_mut);
    let cube_type = prelude_type("CubeType");
    if is_const {
        ty
    } else {
        parse_quote![<#ty as #cube_type>::ExpandType]
    }
}

fn strip_ref(ty: Type, is_ref: &mut bool, is_mut: &mut bool) -> Type {
    match ty {
        Type::Reference(reference) => {
            *is_ref = true;
            *is_mut = *is_mut || reference.mutability.is_some();
            *reference.elem
        }
        ty => ty,
    }
}
