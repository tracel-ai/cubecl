use std::num::NonZero;

use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote, quote_spanned, ToTokens};
use syn::{spanned::Spanned, Ident, Type};

use crate::{expression::Expression, ir_type, prefix_ir};

impl ToTokens for Expression {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let out = match self {
            Expression::Binary {
                left,
                operator,
                right,
                span,
                ..
            } => {
                let span = span.clone();
                let expr_ty = prefix_ir(format_ident!("{}Expr", operator.to_string()));
                quote_spanned! {span=>
                    #expr_ty::new(
                        #left,
                        #right
                    )
                }
            }
            Expression::Unary {
                input,
                operator,
                span,
                ..
            } => {
                let span = span.clone();
                let ty = prefix_ir(format_ident!("{}Expr", operator.to_string()));
                quote_spanned! {span=>
                    #ty::new(
                        #input,
                    )
                }
            }
            Expression::Variable { name, span, .. } => {
                let span = span.clone();
                quote_spanned! {span=>
                    #name.clone()
                }
            }
            Expression::FieldAccess {
                base, field, span, ..
            } => {
                let span = span.clone();
                let access = ir_type("FieldAccess");
                let field = match field {
                    syn::Member::Named(ident) => format_ident!("__{ident}"),
                    syn::Member::Unnamed(index) => format_ident!("__{}", index.index),
                };
                quote_spanned! {span=>
                    #base.expand().#field()
                }
            }
            Expression::Literal { value, span, ty } => {
                let span = span.clone();
                let ir_ty = prefix_ir(format_ident!("Literal"));
                quote_spanned! {span=>
                    #ir_ty {
                        value: #value
                    }
                }
            }
            Expression::Assigment {
                left, right, span, ..
            } => {
                let span = span.clone();
                let ty = prefix_ir(format_ident!("Assignment"));
                quote_spanned! {span=>
                    #ty {
                        left: #left,
                        right: #right
                    }
                }
            }
            Expression::Init {
                left,
                right,
                ty,
                span,
            } => {
                let span = span.clone();
                let ir_type = ir_type("Initializer");
                let ty = right.ty().map(|ty| quote![::<#ty>]);
                quote_spanned! {span=>
                    #ir_type #ty {
                        left: #left,
                        right: #right
                    }
                }
            }
            Expression::Verbatim { tokens } => {
                let span = tokens.span();
                let ty = prefix_ir(format_ident!("Literal"));
                quote_spanned! {span=>
                    #ty {
                        value: #tokens
                    }
                }
            }
            Expression::Block {
                inner,
                ret,
                ty,
                span,
            } => {
                let span = span.clone();
                quote_spanned! {span=>
                    {
                        #(#inner)*
                        #ret
                    }
                }
            }
            Expression::FunctionCall { func, span, args } => {
                let span = span.clone();
                let func = func.as_const().unwrap_or_else(|| quote![#func]);
                // We pass in the `Variable`s and `Literal`s into the expansion so they can be rebound
                // in the function root scope
                quote_spanned! {span=>
                    #func::expand(#(#args),*)
                }
            }
            Expression::MethodCall {
                receiver,
                method,
                args,
                span,
            } => {
                let span = span.clone();
                quote_spanned! {span=>
                    #receiver.expand().#method(#(#args),*)
                }
            }
            Expression::Break { span } => {
                let span = span.clone();
                let brk = ir_type("Break");
                quote_spanned! {span=>
                    #brk
                }
            }
            Expression::Cast { from, to, span } => {
                let span = span.clone();
                let cast = ir_type("Cast");
                quote_spanned! {span=>
                    #cast {
                        from: #from,
                        _to: PhantomData::<#to>
                    }
                }
            }
            Expression::Continue { span } => {
                let span = span.clone();
                let cont = ir_type("Continue");
                quote_spanned! {span=>
                    #cont
                }
            }
            Expression::ForLoop {
                range,
                unroll,
                var_name,
                var_ty,
                var_mut,
                block,
                span,
            } => {
                let span = span.clone();
                let variable = generate_var(
                    var_name,
                    var_ty,
                    span.clone(),
                    Some(quote![::core::num::NonZero::new(1)]),
                );
                let for_ty = ir_type("ForLoop");
                let block_ty = ir_type("Block");
                let block = quote_spanned! {span=>
                    #block_ty::<()>::new(vec![
                        #(#block,)*
                    ])
                };
                quote_spanned! {span=>
                    #for_ty {
                        range: #range,
                        unroll: #unroll,
                        variable: #variable,
                        block: #block,
                    }
                }
            }
            Expression::ConstVariable { name, ty, span } => {
                let span = span.clone();
                let lit_ty = ir_type("Literal");
                quote_spanned! {span=>
                    #lit_ty::new(#name)
                }
            }
        };

        tokens.extend(out);
    }
}

pub fn generate_var(
    name: &Ident,
    ty: &Option<Type>,
    span: Span,
    vectorization: Option<TokenStream>,
) -> TokenStream {
    let var = ir_type("Variable");
    let name = name.to_token_stream().to_string();
    let ty = ty.as_ref().map(|ty| {
        quote_spanned! {ty.span()=>
            ::<#ty>
        }
    });
    let vectorization = vectorization.unwrap_or(quote![None]);
    quote_spanned! {span=>
        #var #ty ::new(#name, #vectorization)
    }
}
