use crate::{
    expression::{Block, Expression},
    paths::{frontend_type, prelude_type},
    scope::Context,
    statement::{Pattern, Statement},
};
use darling::{FromMeta, ast::NestedMeta, util::Flag};
use proc_macro2::{Span, TokenStream};
use quote::{ToTokens, quote};
use std::{collections::HashMap, iter};
use syn::{
    Expr, FnArg, Generics, Ident, ItemFn, LitStr, ReturnType, Signature, TraitItemFn, Type,
    TypeMacro, Visibility, parse, parse_quote, punctuated::Punctuated, spanned::Spanned,
    visit_mut::VisitMut,
};

use super::{desugar::Desugar, helpers::is_comptime_attr, statement::parse_pat};

#[derive(Default, FromMeta)]
pub(crate) struct KernelArgs {
    pub launch: Flag,
    pub launch_unchecked: Flag,
    pub debug_symbols: Flag,
    pub fast_math: Option<Expr>,
    pub debug: Flag,
    pub create_dummy_kernel: Flag,
    pub cluster_dim: Option<Expr>,
    pub src_file: Option<LitStr>,
    /// Base traits for a split expand trait
    pub expand_base_traits: Option<String>,
    /// What self should be taken as for the expansion
    #[darling(default)]
    pub self_type: SelfType,
}

#[derive(Default, FromMeta, PartialEq, Eq)]
pub(crate) enum SelfType {
    #[default]
    Owned,
    Ref,
    RefMut,
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

pub struct GenericAnalysis {
    pub map: HashMap<syn::Ident, syn::PathSegment>,
}

impl GenericAnalysis {
    pub fn process_generics(&self, ty: &syn::Generics) -> TokenStream {
        let mut output = quote![];

        if ty.params.is_empty() {
            return output;
        }

        for param in ty.params.pairs() {
            match param.value() {
                syn::GenericParam::Type(type_param) => {
                    if let Some(ty) = self.map.get(&type_param.ident) {
                        output.extend(quote![#ty,]);
                    } else {
                        let ident = &type_param.ident;
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

    pub fn register_types(&self) -> TokenStream {
        let mut output = quote![];

        for (name, ty) in self.map.iter() {
            output.extend(quote! {
                builder
                    .scope
                    .register_type::<#ty>(#name::as_type_native_unchecked());
            });
        }

        output
    }

    pub fn process_ty(&self, ty: &syn::Type) -> syn::Type {
        let type_path = match &ty {
            Type::Path(type_path) => type_path,
            _ => return ty.clone(),
        };
        let path = &type_path.path;

        let mut returned = syn::Path {
            leading_colon: path.leading_colon,
            segments: syn::punctuated::Punctuated::new(),
        };

        for pair in path.segments.pairs() {
            let segment = pair.value();
            let punc = pair.punct();

            if let Some(segment) = self.map.get(&segment.ident) {
                returned.segments.push_value(segment.clone());
            } else {
                match &segment.arguments {
                    syn::PathArguments::AngleBracketed(arg) => {
                        let mut args = syn::punctuated::Punctuated::new();
                        arg.args.iter().for_each(|arg| match arg {
                            syn::GenericArgument::Type(ty) => {
                                let ty = self.process_ty(ty);
                                args.push(syn::GenericArgument::Type(ty));
                            }
                            _ => args.push_value(arg.clone()),
                        });

                        let segment = syn::PathSegment {
                            ident: segment.ident.clone(),
                            arguments: syn::PathArguments::AngleBracketed(
                                syn::AngleBracketedGenericArguments {
                                    colon2_token: arg.colon2_token,
                                    lt_token: arg.lt_token,
                                    args,
                                    gt_token: arg.gt_token,
                                },
                            ),
                        };
                        returned.segments.push_value(segment);
                    }
                    _ => returned.segments.push_value((*segment).clone()),
                }
            }

            if let Some(punc) = punc {
                returned.segments.push_punct(**punc)
            }
        }

        syn::Type::Path(syn::TypePath {
            qself: type_path.qself.clone(),
            path: returned,
        })
    }

    pub fn from_generics(generics: &syn::Generics) -> Self {
        let mut map = HashMap::new();

        for param in generics.params.pairs() {
            if let syn::GenericParam::Type(type_param) = param.value()
                && let Some(syn::TypeParamBound::Trait(trait_bound)) = type_param.bounds.first()
                && let Some(bound) = trait_bound.path.get_ident()
            {
                let name = bound.to_string();
                let index = map.len() as u8;

                match name.as_str() {
                    "Float" => {
                        map.insert(type_param.ident.clone(), parse_quote!(FloatExpand<#index>));
                    }
                    "Numeric" => {
                        map.insert(
                            type_param.ident.clone(),
                            parse_quote!(NumericExpand<#index>),
                        );
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
    pub analysis: GenericAnalysis,
}

#[derive(Clone)]
pub struct KernelFn {
    pub vis: Visibility,
    pub sig: KernelSignature,
    pub body: KernelBody,
    pub full_name: String,
    pub debug_symbols: bool,
    pub span: Span,
    pub context: Context,
    pub src_file: Option<LitStr>,
}

#[allow(clippy::large_enum_variant)]
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
    pub receiver_arg: Option<FnArg>,
}

impl KernelSignature {
    pub fn runtime_params(&self) -> impl Iterator<Item = &KernelParam> {
        self.parameters.iter().filter(|it| !it.is_const)
    }
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

#[derive(Clone, Debug)]
pub struct KernelParam {
    pub name: Ident,
    pub ty: Type,
    pub normalized_ty: Type,
    pub is_const: bool,
    pub is_mut: bool,
    pub is_ref: bool,
}

impl KernelParam {
    pub fn from_param(param: FnArg, args: &KernelArgs) -> syn::Result<Self> {
        let param = match param {
            FnArg::Typed(param) => param,
            FnArg::Receiver(param) => {
                let mut is_ref = false;
                let mut is_mut = false;
                let normalized_ty =
                    normalize_kernel_ty(*param.ty.clone(), false, &mut is_ref, &mut is_mut);

                let normalized_ty = match args.self_type {
                    SelfType::Owned => normalized_ty,
                    SelfType::Ref => parse_quote!(&#normalized_ty),
                    SelfType::RefMut => parse_quote!(&mut #normalized_ty),
                };

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
    /// Useful when the param is used in functions or methods associated to the
    /// expand type.
    pub fn plain_normalized_self(&mut self) {
        if let Type::Path(pat) = &self.ty
            && pat
                .path
                .get_ident()
                .filter(|ident| *ident == "Self")
                .is_some()
        {
            self.normalized_ty = self.ty.clone();
        }
    }
}

impl KernelSignature {
    pub fn from_signature(sig: Signature, args: &KernelArgs) -> syn::Result<Self> {
        let name = sig.ident;
        let generics = sig.generics;
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
        let parameters = sig
            .inputs
            .into_iter()
            .map(|it| KernelParam::from_param(it, args))
            .collect::<Result<Vec<_>, _>>()?;
        let receiver_arg = if parameters.iter().any(|it| it.name == "self") {
            Some(match args.self_type {
                SelfType::Owned => parse_quote!(self),
                SelfType::Ref => parse_quote!(&self),
                SelfType::RefMut => parse_quote!(&mut self),
            })
        } else {
            None
        };

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
    pub fn plain_returns_self(&mut self) {
        if let Type::Path(pat) = self.returns.ty()
            && pat
                .path
                .get_ident()
                .filter(|ident| *ident == "Self")
                .is_some()
        {
            self.returns = KernelReturns::Plain(self.returns.ty());
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
        let src_file = args.src_file.clone();
        let debug_symbols = args.debug_symbols.is_present();

        let span = Span::call_site();
        let sig = KernelSignature::from_signature(sig, args)?;
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
            src_file,
            context,
            debug_symbols,
        })
    }

    /// We need to call IntoMut::into_mut on mutable owned inputs since their
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
            if !s.is_ref && s.is_mut {
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
        let func = KernelFn::from_sig_and_block(
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
        kernel_generics.params.push(parse_quote![__R: #runtime]);
        let mut expand_generics = kernel_generics.clone();
        expand_generics.params =
            Punctuated::from_iter(iter::once(parse_quote!['kernel]).chain(expand_generics.params));
        let analysis = GenericAnalysis::from_generics(&func.sig.generics);

        Ok(Launch {
            args,
            vis,
            func,
            kernel_generics,
            launch_generics: expand_generics,
            analysis,
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

pub fn strip_ref(ty: Type, is_ref: &mut bool, is_mut: &mut bool) -> Type {
    match ty {
        Type::Reference(reference) => {
            *is_ref = true;
            *is_mut = *is_mut || reference.mutability.is_some();
            *reference.elem
        }
        ty => ty,
    }
}
