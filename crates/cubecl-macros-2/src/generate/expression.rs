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
                let binop = ir_type("BinaryOp");
                quote_spanned! {span=>
                    #expr_ty(#binop::new(
                        Box::new(#left),
                        Box::new(#right)
                    ))
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
                let ty_un = prefix_ir(format_ident!("UnaryOp"));
                quote_spanned! {span=>
                    #ty(#ty_un::new(
                        Box::new(#input),
                    ))
                }
            }
            Expression::Variable { name, span, ty } => {
                let span = span.clone();
                quote_spanned! {span=>
                    #name.clone()
                }
            }
            Expression::FieldAccess {
                base,
                field,
                span,
                struct_ty,
            } => {
                let span = span.clone();
                let access = ir_type("FieldAccess");
                let kernel_struct = ir_type("KernelStruct");
                quote_spanned! {span=>
                    <#struct_ty as #kernel_struct>::expand(#base).#field
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
                // TODO: Make expand return Block<T>
                // We pass in the `Variable`s and `Literal`s into the expansion so they can be rebound
                // in the function root scope
                quote_spanned! {span=>
                    #func ::expand(#(#args.into()),*)
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
                from,
                to,
                step,
                unroll,
                var_name,
                var_ty,
                var_mut,
                block,
                span,
            } => {
                let span = span.clone();
                let variable = generate_var(var_name, var_ty, span.clone());
                let for_ty = ir_type("ForLoop");
                let block_ty = ir_type("Block");
                let step = if let Some(step) = step {
                    quote![Some(Box::new(#step))]
                } else {
                    quote![None]
                };
                let block = quote_spanned! {span=>
                    #block_ty::<()> {
                        statements: vec![
                            #(#block,)*
                        ],
                        _ty: ::core::marker::PhantomData
                    }
                };
                quote_spanned! {span=>
                    #for_ty {
                        from: Box::new(#from),
                        to: Box::new(#to),
                        step: #step,
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

pub fn generate_var(name: &Ident, ty: &Option<Type>, span: Span) -> TokenStream {
    let var = ir_type("Variable");
    let name = name.to_token_stream().to_string();
    let ty = ty.as_ref().map(|ty| {
        quote_spanned! {ty.span()=>
            ::<#ty>
        }
    });
    quote_spanned! {span=>
        #var #ty {
            name: #name,
            _type: ::core::marker::PhantomData
        }
    }
}
