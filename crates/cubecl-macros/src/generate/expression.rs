use cubecl_common::operator::Operator;
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote, quote_spanned, ToTokens};
use syn::{spanned::Spanned, Ident, PathArguments, Type};

use crate::{
    expression::{Block, Expression},
    generate::kernel::CONTEXT,
    paths::{frontend_path, frontend_type, prelude_type},
};

macro_rules! error {
    ($span:expr, $fmt:literal $(,$args:expr)*) => {
        syn::Error::new($span, format!($fmt $(,$args)*)).into_compile_error()
    };
}

impl ToTokens for Expression {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let out = match self {
            Expression::Binary {
                left,
                operator,
                right,
                span,
                ..
            } if operator.is_assign() && matches!(**left, Expression::Index { .. }) => {
                let frontend_path = frontend_path();
                let (array, index) = left.as_index().unwrap();
                let op = format_ident!("{}", operator.array_op_name());
                quote_spanned! {*span=>
                    {
                        let _array = #array;
                        let _index = #index;
                        let _value = #right;
                        #frontend_path::#op::expand(context, _array, _index, _value)
                    }
                }
            }
            Expression::Binary {
                left,
                operator,
                right,
                span,
                ..
            } => {
                let frontend_path = frontend_path();
                let op = format_ident!("{}", operator.op_name());
                quote_spanned! {*span=>
                    {
                        let _lhs = #left;
                        let _rhs = #right;
                        #frontend_path::#op::expand(context, _lhs, _rhs)
                    }
                }
            }
            Expression::Unary {
                input,
                operator: Operator::Not,
                span,
                ..
            } => {
                let frontend_path = frontend_path();
                quote_spanned! {*span=>
                    {
                        let _inner = #input;
                        #frontend_path::not::expand(context, _inner)
                    }
                }
            }
            Expression::Unary {
                input,
                operator: Operator::Deref,
                ..
            } => quote![#input],
            Expression::Unary { operator, span, .. } => {
                error!(*span, "Unary operator {operator} not yet supported")
            }
            Expression::Keyword { name } => {
                quote![#name::expand(context)]
            }
            Expression::Variable { name, .. } => {
                let last_use = CONTEXT.with_borrow(|ctx| ctx.try_consume(name));
                if last_use {
                    quote![#name]
                } else {
                    quote![#name.clone()]
                }
            }
            Expression::FieldAccess {
                base, field, span, ..
            } => {
                let field = match field {
                    syn::Member::Named(ident) => format_ident!("__{ident}"),
                    syn::Member::Unnamed(index) => format_ident!("__{}", index.index),
                };
                quote_spanned! {*span=>
                    #base.#field.clone()
                }
            }
            Expression::Literal { value, .. } => {
                let expand_elem = frontend_type("ExpandElementTyped");
                quote![#expand_elem::from_lit(#value)]
            }
            Expression::Assigment {
                left, right, span, ..
            } if matches!(**left, Expression::Index { .. }) => {
                let (array, index) = left.as_index().unwrap();
                let frontend_path = frontend_path();
                quote_spanned! {*span=>
                    let _array = #array;
                    let _index = #index;
                    let _value = #right;
                    #frontend_path::index_assign::expand(context, _array, _index, _value)
                }
            }
            Expression::Assigment {
                left, right, span, ..
            } => {
                let frontend_path = frontend_path();
                quote_spanned! {*span=>
                    let _var = #left;
                    let _value = #right;
                    #frontend_path::assign::expand(context, _value, _var)
                }
            }
            Expression::Verbatim { tokens, .. } => {
                let span = tokens.span();
                quote_spanned! {span=>
                    #tokens
                }
            }
            Expression::Block(block) => block.to_token_stream(),
            Expression::FunctionCall {
                func,
                span,
                args,
                associated_type,
            } => {
                let args: Vec<TokenStream> = if self.is_const() {
                    args.iter().map(|arg| arg.to_token_stream()).collect()
                } else {
                    let once_expr = frontend_type("OnceExpr");
                    args.iter()
                        .map(|arg| {
                            if arg.is_const() {
                                arg.to_token_stream()
                            } else {
                                quote![#once_expr::new(#arg)]
                            }
                        })
                        .collect()
                };

                // We pass in the `Variable`s and `Literal`s into the expansion so they can be rebound
                // in the function root scope
                if let Some((ty_path, name)) = associated_type {
                    let static_expand = frontend_type("StaticExpand");
                    quote_spanned! {*span=>
                        <#ty_path as #static_expand>::Expanded::#name(#(#args),*)
                    }
                } else {
                    let (generics, path) = split_generics(func);
                    quote_spanned! {*span=>
                        #path::expand #generics(#(#args),*)
                    }
                }
            }
            Expression::MethodCall {
                receiver,
                method,
                generics,
                args,
                span,
            } => {
                let method = format_ident!("__expand_{method}_method");
                quote_spanned! {*span=>
                    #receiver.#method #generics(#(#args),*)
                }
            }
            Expression::Break { span } => {
                let path = frontend_path();
                quote_spanned! {*span=>
                    #path::branch::break_expand(context);
                }
            }
            Expression::Continue { span } => error!(*span, "Continue not supported yet"),
            Expression::Return { expr, span, .. } => {
                if expr.is_some() {
                    error!(*span, "Only void return is supported.")
                } else {
                    quote::quote! {
                        cubecl::frontend::branch::return_expand(context);
                    }
                }
            }
            Expression::Cast { from, to, span } => {
                let cast = prelude_type("Cast");
                quote_spanned! {*span=>
                    <#to as #cast>::cast_from(#from)
                }
            }
            Expression::ForLoop {
                range,
                unroll,
                var_name,
                var_ty,
                block,
                span,
            } => {
                let variable = generate_var(var_name, true, var_ty, *span, None);
                let for_ty = frontend_type("ForLoop");

                if let Some(unroll) = unroll {
                    //let unrolled = generate_unroll(block, range, var_name);
                    quote_spanned! {*span=>
                        {
                            let #var_name = #variable;
                            if #unroll {
                                #for_ty::new_unroll(#range, #var_name.clone(), #block)
                            } else {
                                #for_ty::new(#range, #var_name.clone(), #block)
                            }
                        }
                    }
                } else {
                    quote_spanned! {*span=>
                        {
                            let #var_name = #variable;
                            #for_ty::new(#range, #var_name.clone(), #block)
                        }
                    }
                }
            }
            Expression::WhileLoop {
                condition,
                block,
                span,
            } => {
                let while_ty = frontend_type("WhileLoop");

                quote_spanned! {*span=>
                    {
                        #while_ty::new(#condition, #block)
                    }
                }
            }
            Expression::Loop { block, span } => {
                let loop_ty = frontend_type("Loop");

                quote_spanned! {*span=>
                    {
                        #loop_ty::new(#block)
                    }
                }
            }
            Expression::If {
                condition,
                then_block,
                else_branch,
                span,
            } if condition.is_const() => {
                let as_const = condition.as_const().unwrap();
                let else_branch = else_branch.as_ref().map(|it| quote![else #it]);
                quote_spanned! {*span=>
                    if #as_const #then_block #else_branch
                }
            }
            Expression::If {
                condition,
                then_block,
                else_branch: Some(else_branch),
                span,
            } => {
                let path = frontend_path();
                quote_spanned! {*span=>
                    let _cond = #condition;
                    #path::branch::if_else_expand(context, None, _cond.into(), |context| #then_block, |context| #else_branch);
                }
            }
            Expression::If {
                condition,
                then_block,
                span,
                ..
            } => {
                let path = frontend_path();
                quote_spanned! {*span=>
                    let _cond = #condition;
                    #path::branch::if_expand(context, None, _cond.into(), |context| #then_block);
                }
            }
            Expression::ConstVariable { name, .. } => quote![#name],
            Expression::Path { path, .. } => quote![#path],
            Expression::Range {
                start,
                end,
                inclusive,
                span,
            } => {
                if let Some(end) = end {
                    let range = frontend_type("RangeExpr");
                    quote_spanned! {*span=>
                        #range::new(#start, #end, #inclusive)
                    }
                } else {
                    let range = frontend_type("SliceRangeExpr");
                    let end = end
                        .as_ref()
                        .map(|it| quote![Some(Box::new(#it))])
                        .unwrap_or_else(|| quote![None]);
                    quote_spanned! {*span=>
                        #range::new(#start, #end, #inclusive)
                    }
                }
            }

            Expression::Array { span, .. } => {
                if let Some(constant) = self.as_const() {
                    constant
                } else {
                    syn::Error::new(*span, "Array expressions can't be used at runtime")
                        .to_compile_error()
                }
            }
            Expression::Tuple { span, .. } => {
                if let Some(constant) = self.as_const() {
                    constant
                } else {
                    syn::Error::new(*span, "Tuple expressions can't be used at runtime")
                        .to_compile_error()
                }
            }
            Expression::Index { expr, index, span } => {
                quote_spanned! {*span=>
                    #expr.expand().index(#index)
                }
            }
            Expression::Slice { expr, ranges, span } => {
                let range_ty = frontend_type("SliceRangeExpr");
                quote_spanned! {*span=>
                    #expr.expand().slice(vec![#(Box::new(#range_ty::from(#ranges))),*])
                }
            }
            Expression::ArrayInit { init, len, span } => {
                let init_ty = frontend_type("ArrayInit");
                quote_spanned! {*span=>
                    #init_ty::new(#len, #init)
                }
            }
            Expression::VerbatimTerminated { tokens } => tokens.clone(),
            Expression::Reference { inner } => {
                if let Some(as_const) = inner.as_const() {
                    quote![&#as_const]
                } else {
                    quote![#inner]
                }
            }
            Expression::StructInit { path, fields } => {
                let cube_type = frontend_type("CubeType");

                quote! {
                    <#path as #cube_type>::Runtime::new(#(#fields),*)
                }
            }
            Expression::Closure { tokens } => tokens.clone(),
        };

        tokens.extend(out);
    }
}

impl ToTokens for Block {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        CONTEXT.with_borrow_mut(|ctx| ctx.restore_scope());
        let ret = self
            .ret
            .as_ref()
            .map(|ret| quote![#ret])
            .unwrap_or_else(|| quote![()]);
        let inner = &self.inner;
        tokens.extend(quote_spanned! {self.span=>
            {
                #(#inner)*
                #ret
            }
        });
        CONTEXT.with_borrow_mut(|ctx| ctx.delete_scope());
    }
}

pub fn generate_var(
    name: &Ident,
    mutable: bool,
    ty: &Option<Type>,
    span: Span,
    vectorization: Option<TokenStream>,
) -> TokenStream {
    let var = frontend_type("Variable");
    let name = name.to_token_stream().to_string();
    let ty = ty.as_ref().map(|ty| {
        quote_spanned! {ty.span()=>
            ::<#ty>
        }
    });
    let vectorization = vectorization.unwrap_or(quote![None]);
    quote_spanned! {span=>
        #var #ty ::new(#name, #mutable, #vectorization)
    }
}

fn split_generics(path: &Expression) -> (PathArguments, TokenStream) {
    let mut path = match path {
        Expression::Path { path, .. } => path.clone(),
        _ => return (PathArguments::None, quote![#path]),
    };
    let generics = if let Some(last) = path.segments.last_mut() {
        core::mem::replace(&mut last.arguments, PathArguments::None)
    } else {
        PathArguments::None
    };
    (generics, quote![#path])
}
