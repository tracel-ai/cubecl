use crate::{expression::Block, paths::prelude_type, scope::Context, statement::Pattern};
use darling::{ast::NestedMeta, util::Flag, FromMeta};
use proc_macro2::TokenStream;
use std::iter;
use syn::{
    parse_quote, punctuated::Punctuated, spanned::Spanned, visit_mut::VisitMut, Expr, FnArg,
    Generics, Ident, ItemFn, Signature, TraitItemFn, Type, Visibility,
};

use super::{desugar::Desugar, helpers::is_comptime_attr, statement::parse_pat};

#[derive(Default, FromMeta)]
pub(crate) struct KernelArgs {
    pub launch: Flag,
    pub launch_unchecked: Flag,
    pub debug: Flag,
    pub create_dummy_kernel: Flag,
    pub local_allocator: Option<Expr>,
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
    pub body: KernelBody,
    pub context: Context,
}

#[derive(Clone)]
pub enum KernelBody {
    Block(Block),
    Verbatim(TokenStream),
}

#[derive(Clone)]
pub struct KernelSignature {
    pub name: Ident,
    pub parameters: Vec<KernelParam>,
    pub returns: KernelReturns,
    pub generics: Generics,
}

#[derive(Clone)]
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
            FnArg::Receiver(param) => {
                let mut is_ref = false;
                let mut is_mut = false;
                let normalized_ty =
                    normalize_kernel_ty(*param.ty.clone(), false, &mut is_ref, &mut is_mut);

                is_mut = param.mutability.is_some();
                is_ref = param.reference.is_some();

                return Ok(KernelParam {
                    name: Ident::new("self", param.span()),
                    ty: *param.ty,
                    normalized_ty,
                    is_const: false,
                    is_mut,
                    is_ref,
                });
            }
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

    /// If the type is self, we set the normalized ty to self as well.
    ///
    /// Useful when the param is used in functions or methods associated to the expand type.
    pub fn plain_normalized_self(&mut self) {
        if let Type::Path(pat) = &self.ty {
            if pat
                .path
                .get_ident()
                .filter(|ident| *ident == "Self")
                .is_some()
            {
                self.normalized_ty = self.ty.clone();
            }
        }
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
            returns: KernelReturns::ExpandType(returns),
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
            returns: KernelReturns::ExpandType(returns),
        })
    }

    /// If the type is self, we set the returns type to plain instead of expand type.
    pub fn plain_returns_self(&mut self) {
        if let Type::Path(pat) = self.returns.ty() {
            if pat
                .path
                .get_ident()
                .filter(|ident| *ident == "Self")
                .is_some()
            {
                self.returns = KernelReturns::Plain(self.returns.ty());
            }
        }
    }
}

impl KernelFn {
    pub fn from_sig_and_block(sig: Signature, mut block: syn::Block) -> syn::Result<Self> {
        let sig = KernelSignature::from_signature(sig)?;
        Desugar.visit_block_mut(&mut block);

        let mut context = Context::new(sig.returns.ty());
        context.extend(sig.parameters.clone());
        let (block, _) = context.in_scope(|ctx| Block::from_block(block, ctx))?;

        Ok(KernelFn {
            sig,
            body: KernelBody::Block(block),
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
