use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::{ExprClosure, Type};

use crate::parse::const_eval::{ConstEval, ConstEvalArm, CustomArm, FoldArm};

impl ConstEvalArm {
    pub fn generate_arm(&self) -> syn::Result<Vec<TokenStream>> {
        let mut tokens = vec![];
        for attr_ty in self.attr_types.0.iter() {
            if attr_ty.subtypes.is_empty() {
                tokens.push(self.generate_attr(&attr_ty.attr_ty, &self.closure, None)?);
            } else {
                for ty in attr_ty.subtypes.iter() {
                    tokens.push(self.generate_attr(&attr_ty.attr_ty, &self.closure, Some(ty))?);
                }
            }
        }
        Ok(tokens)
    }

    fn generate_attr(
        &self,
        attr_ty: &Type,
        closure: &ExprClosure,
        ty: Option<&Type>,
    ) -> syn::Result<TokenStream> {
        if closure.inputs.is_empty() {
            return Err(syn::Error::new_spanned(
                closure.inputs.clone(),
                "Expected at least one input for constant folding",
            ));
        }
        let return_ty = match &closure.output {
            syn::ReturnType::Default => None,
            syn::ReturnType::Type(_, ty) => Some(ty),
        };
        let attr_defs = (0..closure.inputs.len()).map(|i| {
            let name = format_ident!("attr_{i}");
            quote![let #name = operand_attrs[#i].as_ref()?.downcast_ref::<#attr_ty>()?;]
        });
        let value_defs = closure.inputs.iter().enumerate().map(|(i, name)| {
            let attr_name = format_ident!("attr_{i}");
            if let Some(ty) = ty {
                quote![let #name = #attr_name.value::<#ty>(ctx)?;]
            } else {
                quote![let #name = #attr_name.value()?;]
            }
        });

        let body = &closure.body;
        let with_value = if let Some(return_ty) = return_ty {
            quote![{
                #[allow(unused_braces)]
                let __x: #return_ty = #body;
                __x
            }]
        } else if let Some(ty) = ty {
            quote![attr_0.with_value::<#ty>(#body)]
        } else {
            quote![attr_0.with_value(#body)]
        };

        Ok(quote! {
            (|| {
                #(#attr_defs)*
                #(#value_defs)*
                Some(pliron::attribute::AttrObj::from(#with_value))
            })()
        })
    }
}

impl CustomArm {
    pub fn generate_arm(&self) -> syn::Result<TokenStream> {
        let closure = &self.closure;
        if closure.inputs.is_empty() {
            return Err(syn::Error::new_spanned(
                closure.inputs.clone(),
                "Expected at least one input for constant folding",
            ));
        }
        let attr_defs = closure.inputs.iter().enumerate().map(|(i, name)| {
            quote! {
                let #name = if let Some(attr) = operand_attrs[#i].as_ref() {
                    Some(attr_cast::<dyn crate::interfaces::ConstantAttr>(&**attr)?)
                } else {
                    None
                };
            }
        });

        let body = &closure.body;

        Ok(quote! {
            (|| {
                #(#attr_defs)*
                #[allow(unused_braces)]
                Some(pliron::attribute::AttrObj::from({
                    #body
                }?))
            })()
        })
    }
}

impl ConstEval {
    pub fn generate_arms(&self) -> syn::Result<TokenStream> {
        if self.arms.is_empty() {
            return Err(syn::Error::new(
                Span::call_site(),
                "Expected at least one attribute arm for constant folding",
            ));
        }
        let arms = self.arms.iter().map(|it| match it {
            FoldArm::ConstEval(it) => it.generate_arm(),
            FoldArm::Custom(it) => Ok(vec![it.generate_arm()?]),
        });
        let mut arms = arms.collect::<Result<Vec<_>, _>>()?.into_iter().flatten();
        let first = arms.next().unwrap();
        Ok(arms.fold(first, |acc, arm| quote![#acc.or(#arm)]))
    }
}

pub fn generate_const_eval(input: TokenStream) -> syn::Result<TokenStream> {
    let const_eval: ConstEval = syn::parse2(input)?;

    let ty = &const_eval.op_type;
    let arms = const_eval.generate_arms()?;

    Ok(quote! {
        #[pliron::derive::op_interface_impl]
        impl pliron::opts::constants::ConstFoldInterface for #ty {
            fn check_fold(
                &self,
                ctx: &pliron::context::Context,
                operand_attrs: &[Option<pliron::attribute::AttrObj>],
            ) -> alloc::vec::Vec<Option<pliron::attribute::AttrObj>> {
                use crate::interfaces::ConstantAttr;
                use pliron::builtin::attr_interfaces::TypedAttrInterface;
                use pliron::attribute::attr_cast;
                alloc::vec![#arms]
            }

            fn fold_in_place(
                &self,
                ctx: &mut pliron::context::Context,
                operand_attrs: &[Option<pliron::attribute::AttrObj>],
                rewriter: &mut dyn pliron::irbuild::rewriter::Rewriter,
            ) -> pliron::irbuild::IRStatus {
                self.fold_with_materialization(ctx, operand_attrs, rewriter)
            }
        }

    })
}
