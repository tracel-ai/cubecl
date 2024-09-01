use darling::{ast::NestedMeta, util::Flag, FromMeta};
use proc_macro2::{Span, TokenStream};
use syn::{
    parse_quote, spanned::Spanned, Block, FnArg, Generics, Ident, ItemFn, Path, Signature,
    TraitItemFn, Type, Visibility,
};

use crate::{expression::Expression, ir_type, scope::Context, statement::parse_pat};

use super::{branch::parse_block, helpers::is_comptime_attr};

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

#[derive(Default, FromMeta)]
pub(crate) struct CubeTraitArgs {
    pub expand_name: Option<Ident>,
}

#[derive(Default, FromMeta)]
pub(crate) struct CubeTraitImplArgs {
    pub expand_name: Option<Ident>,
    pub trait_expand_name: Option<Path>,
}

impl KernelArgs {
    pub fn is_launch(&self) -> bool {
        self.launch.is_present() || self.launch_unchecked.is_present()
    }
}

pub struct Kernel {
    pub args: KernelArgs,
    pub vis: Visibility,
    pub func: KernelFn,
}

#[derive(Clone)]
pub struct KernelFn {
    pub sig: KernelSignature,
    pub kernel_vars: Vec<TokenStream>,
    pub block: Expression,
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
    pub span: Span,
}

impl KernelParam {
    fn from_param(param: FnArg) -> syn::Result<Self> {
        let span = param.span();
        let param = match param {
            FnArg::Typed(param) => param,
            param => Err(syn::Error::new_spanned(
                param,
                "Can't use `cube` on methods",
            ))?,
        };
        let (name, _, mut mutable) = parse_pat(*param.pat)?;
        let is_const = param.attrs.iter().any(is_comptime_attr);
        let ty = *param.ty.clone();
        let normalized_ty = normalize_kernel_ty(*param.ty, is_const, &mut mutable);
        Ok(Self {
            name,
            ty,
            normalized_ty,
            is_const,
            is_mut: mutable,
            span,
        })
    }

    pub fn ty_owned(&self) -> Type {
        strip_ref(self.ty.clone(), &mut false)
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
    pub fn from_sig_and_block(sig: Signature, block: Block, launch: bool) -> syn::Result<Self> {
        let sig = KernelSignature::from_signature(sig)?;

        let mut context = Context::new(sig.returns.clone(), launch);
        let kernel_vars = context.current_scope().generate_kernel_vars();
        context.extend(sig.parameters.clone());
        context.push_scope(); // Push function local scope
        let block = parse_block(block, &mut context)?;
        context.pop_scope(); // Pop function local scope

        Ok(KernelFn {
            sig,
            block,
            kernel_vars,
        })
    }
}

impl Kernel {
    pub fn from_item_fn(function: ItemFn, args: KernelArgs) -> syn::Result<Self> {
        let vis = function.vis;
        let func = KernelFn::from_sig_and_block(function.sig, *function.block, args.is_launch())?;

        Ok(Kernel { args, vis, func })
    }
}

fn normalize_kernel_ty(ty: Type, is_const: bool, is_ref_mut: &mut bool) -> Type {
    let ty = strip_ref(ty, is_ref_mut);
    let expr = ir_type("Expr");
    if is_const {
        ty
    } else {
        parse_quote![impl #expr<Output = #ty> + 'static + Clone]
    }
}

fn strip_ref(ty: Type, is_ref_mut: &mut bool) -> Type {
    match ty {
        Type::Reference(reference) => {
            *is_ref_mut = *is_ref_mut || reference.mutability.is_some();
            *reference.elem
        }
        ty => ty,
    }
}
