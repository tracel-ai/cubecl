use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::ExprClosure;

use crate::parse::simplify::Simplify;

impl Simplify {
    pub fn generate_arms(&self) -> syn::Result<TokenStream> {
        if self.arms.is_empty() {
            return Err(syn::Error::new(
                Span::call_site(),
                "Expected at least one attribute arm for simplification",
            ));
        }
        let arms = self.arms.iter().map(|it| self.generate_arm(it));
        let mut arms = arms.collect::<Result<Vec<_>, _>>()?.into_iter();
        let first = arms.next().unwrap();
        Ok(arms.fold(first, |acc, arm| quote![#acc.or(#arm)]))
    }

    fn generate_arm(&self, closure: &ExprClosure) -> syn::Result<TokenStream> {
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
                #body
            })()
        })
    }
}

pub fn generate_simplify(input: TokenStream) -> syn::Result<TokenStream> {
    let const_eval: Simplify = syn::parse2(input)?;

    let ty = &const_eval.op_type;
    let arms = const_eval.generate_arms()?;

    Ok(quote! {
        #[pliron::derive::op_interface_impl]
        impl crate::interfaces::SimplifyInterface for #ty {
            fn check_fold(
                &self,
                ctx: &pliron::context::Context,
                operand_attrs: &[Option<pliron::attribute::AttrObj>],
            ) -> Option<pliron::value::Value> {
                use crate::interfaces::ConstantAttr;
                use pliron::builtin::attr_interfaces::TypedAttrInterface;
                use pliron::attribute::attr_cast;
                #arms
            }
        }

    })
}
