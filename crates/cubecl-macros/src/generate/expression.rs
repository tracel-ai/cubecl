use std::mem;

use proc_macro2::TokenStream;
use quote::{format_ident, quote, quote_spanned};
use syn::{spanned::Spanned, PathArguments};

use crate::{
    expression::{Block, Expression},
    operator::Operator,
    paths::{frontend_path, frontend_type, prelude_type},
    scope::Context,
};

macro_rules! error {
    ($span:expr, $fmt:literal $(,$args:expr)*) => {
        syn::Error::new($span, format!($fmt $(,$args)*)).into_compile_error()
    };
}

impl Expression {
    pub fn to_tokens(&self, context: &mut Context) -> TokenStream {
        match self {
            Expression::Binary {
                left,
                operator,
                right,
                span,
                ..
            } if operator.is_assign() && matches!(**left, Expression::Index { .. }) => {
                let frontend_path = frontend_path();
                let (array, index) = left.as_index().unwrap();
                let array = array.to_tokens(context);
                let index = index.to_tokens(context);
                let right = right.to_tokens(context);
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
                let left = left.to_tokens(context);
                let right = right.to_tokens(context);
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
                let input = input.to_tokens(context);
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
            } => input.to_tokens(context),
            Expression::Unary { operator, span, .. } => {
                error!(*span, "Unary operator {operator} not yet supported")
            }
            Expression::Keyword { name } => {
                quote![#name::expand(context)]
            }
            Expression::Variable { name, .. } => {
                let last_use = context.try_consume(name);
                if last_use {
                    quote![#name]
                } else {
                    quote![#name.clone()]
                }
            }
            Expression::FieldAccess {
                base, field, span, ..
            } => {
                let base = base.to_tokens(context);
                quote_spanned! {*span=>
                    #base.#field.clone()
                }
            }
            Expression::Literal { value, .. } => {
                let expand_elem = frontend_type("ExpandElementTyped");
                quote![#expand_elem::from_lit(#value)]
            }
            Expression::ConstVariable { name, .. } => {
                let expand_elem = frontend_type("ExpandElementTyped");
                quote![#expand_elem::from_lit(#name)]
            }
            Expression::Assigment {
                left, right, span, ..
            } if matches!(**left, Expression::Index { .. }) => {
                let (array, index) = left.as_index().unwrap();
                let array = array.to_tokens(context);
                let index = index.to_tokens(context);
                let right = right.to_tokens(context);
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
                let left = left.to_tokens(context);
                let right = right.to_tokens(context);
                quote_spanned! {*span=>
                    let _var = #left;
                    let _value = #right;
                    #frontend_path::assign::expand(context, _value, _var)
                }
            }
            Expression::Index { expr, index, span } => {
                let expr = expr.to_tokens(context);
                let index = index.to_tokens(context);
                let index_fn = frontend_type("index");
                quote_spanned! {*span=>
                    {
                        let _array = #expr;
                        let _index = #index;
                        #index_fn::expand(context, _array, _index)
                    }
                }
            }
            Expression::FunctionCall {
                func,
                span,
                args,
                associated_type: None,
            } => {
                let (args, arg_names) = map_args(args, context);
                let (generics, path) = split_generics(func, context);
                quote_spanned! {*span=>
                    {
                        #(#args)*
                        #path::expand #generics(context, #(#arg_names),*)
                    }
                }
            }
            Expression::FunctionCall {
                span,
                args,
                associated_type: Some((ty_path, func)),
                ..
            } => {
                let (args, arg_names) = map_args(args, context);
                let mut name = func.clone();
                name.ident = format_ident!("__expand_{}", name.ident);
                quote_spanned! {*span=>
                    {
                        #(#args)*
                        #ty_path::#name(context, #(#arg_names),*)
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
                let receiver = receiver
                    .as_const(context)
                    .unwrap_or_else(|| receiver.to_tokens(context));
                let (args, arg_names) = map_args(args, context);
                quote_spanned! {*span=>
                    {
                        #(#args)*
                        #receiver.#method #generics(context, #(#arg_names),*)
                    }
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
                let from = from.to_tokens(context);
                quote_spanned! {*span=>
                    <#to as #cast>::__expand_cast_from(context, #from)
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
                let for_ty = frontend_type("branch");

                let range = range.to_tokens(context);
                let unroll = unroll
                    .as_ref()
                    .and_then(|it| it.as_const(context))
                    .unwrap_or(quote![false]);
                let must_clone = context.must_clone;
                context.must_clone = true;
                let block = block.to_tokens(context);
                context.must_clone = must_clone;
                let var_ty = var_ty.as_ref().map(|it| quote![: #it]);

                quote_spanned! {*span=>
                    {
                        let _range = #range;
                        let _unroll = #unroll;
                        #for_ty::for_expand(context, _range, _unroll, |context, #var_name #var_ty| #block);
                    }
                }
            }
            Expression::WhileLoop {
                condition,
                block,
                span,
            } => {
                let while_ty = frontend_type("branch");
                let condition = condition.to_tokens(context);
                let block = block.to_tokens(context);

                quote_spanned! {*span=>
                    {
                        #while_ty::while_loop_expand(context, |context| #condition, |context| #block);
                    }
                }
            }
            Expression::Loop { block, span } => {
                let loop_ty = frontend_type("branch");
                let block = block.to_tokens(context);

                quote_spanned! {*span=>
                    #loop_ty::loop_expand(context, |context| #block);
                }
            }
            Expression::If {
                condition,
                then_block,
                else_branch,
                span,
            } if condition.is_const() => {
                let as_const = condition.as_const(context).unwrap();
                let then_block = then_block.to_tokens(context);
                let else_branch = else_branch
                    .as_ref()
                    .map(|it| it.to_tokens(context))
                    .map(|it| quote![else #it]);
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
                let condition = condition.to_tokens(context);
                let must_clone = mem::replace(&mut context.must_clone, true);
                let then_block = then_block.to_tokens(context);
                let else_branch = else_branch.to_tokens(context);
                context.must_clone = must_clone;
                quote_spanned! {*span=>
                    let _cond = #condition;
                    #path::branch::if_else_expand(context, _cond.into(), |context| #then_block, |context| #else_branch);
                }
            }
            Expression::If {
                condition,
                then_block,
                span,
                ..
            } => {
                let path = frontend_path();
                let condition = condition.to_tokens(context);
                let then_block = then_block.to_tokens(context);
                quote_spanned! {*span=>
                    let _cond = #condition;
                    #path::branch::if_expand(context, _cond.into(), |context| #then_block);
                }
            }
            Expression::Path { path, .. } => quote![#path],
            Expression::Range {
                start,
                end,
                inclusive,
                span,
            } => {
                let start = start
                    .as_const(context)
                    .unwrap_or_else(|| start.to_tokens(context));
                if let Some(end) = end {
                    let range = frontend_type("Range");
                    let end = end
                        .as_const(context)
                        .unwrap_or_else(|| end.to_tokens(context));
                    quote_spanned! {*span=>
                        {
                            let _start = #start;
                            let _end = #end;
                            #range::new(_start.into(), _end.into(), #inclusive)
                        }
                    }
                } else {
                    error!(*span, "Slice range not yet supported")
                    // let range = frontend_type("SliceRangeExpr");
                    // quote_spanned! {*span=>
                    //     #range::new(#start, None, #inclusive)
                    // }
                }
            }

            Expression::Array { span, .. } => {
                if let Some(constant) = self.as_const(context) {
                    constant
                } else {
                    syn::Error::new(*span, "Array expressions can't be used at runtime")
                        .to_compile_error()
                }
            }
            Expression::Tuple { span, .. } => {
                if let Some(constant) = self.as_const(context) {
                    constant
                } else {
                    syn::Error::new(*span, "Tuple expressions can't be used at runtime")
                        .to_compile_error()
                }
            }

            Expression::Slice { expr, ranges, span } => {
                let range_ty = frontend_type("SliceRangeExpr");
                let expr = expr.to_tokens(context);
                let ranges = ranges.iter().map(|it| it.to_tokens(context));

                quote_spanned! {*span=>
                    #expr.expand().slice(vec![#(Box::new(#range_ty::from(#ranges))),*])
                }
            }
            Expression::ArrayInit { init, len, span } => {
                let init_ty = frontend_type("ArrayInit");
                let init = init.to_tokens(context);
                let len = len.to_tokens(context);

                quote_spanned! {*span=>
                    #init_ty::new(#len, #init)
                }
            }
            Expression::VerbatimTerminated { tokens } => tokens.clone(),
            Expression::Reference { inner } => {
                if let Some(as_const) = inner.as_const(context) {
                    quote![&#as_const]
                } else {
                    let inner = inner.to_tokens(context);
                    quote![#inner]
                }
            }
            Expression::StructInit { path, fields } => {
                let cube_type = prelude_type("CubeType");
                let fields = fields.iter().map(|(pat, it)| {
                    let value = it
                        .as_const(context)
                        .map(|as_const| quote![#as_const.into()])
                        .unwrap_or_else(|| it.to_tokens(context));
                    quote![#pat: #value]
                });
                let path_last = path.segments.last().unwrap();
                let turbofish = path_last.arguments.clone();
                let generics = match &turbofish {
                    PathArguments::None => None,
                    PathArguments::AngleBracketed(params) => {
                        let params = params.args.iter();
                        Some(quote![<#(#params),*>])
                    }
                    _ => panic!("Fn generics not supported when constructing runtime structs"),
                };

                quote! {
                    {
                        type _Ty #generics = <#path as #cube_type>::ExpandType;
                        _Ty #turbofish { #(#fields),* }
                    }
                }
            }
            Expression::Closure { tokens } => tokens.clone(),
            Expression::Verbatim { tokens, .. } => tokens.clone(),
            Expression::Block(block) => block.to_tokens(context),
        }
    }
}

impl Block {
    pub fn to_tokens(&self, context: &mut Context) -> TokenStream {
        context.restore_scope();

        let inner: Vec<_> = self.inner.iter().map(|it| it.to_tokens(context)).collect();
        let ret = self
            .ret
            .as_ref()
            .map(|ret| ret.to_tokens(context))
            .unwrap_or_else(|| quote![()]);

        context.delete_scope();
        quote_spanned! {self.span=>
            {
                #(#inner)*
                #ret
            }
        }
    }
}

fn split_generics(path: &Expression, context: &mut Context) -> (PathArguments, TokenStream) {
    let mut path = match path {
        Expression::Path { path, .. } => path.clone(),
        _ => return (PathArguments::None, path.to_tokens(context)),
    };
    let generics = if let Some(last) = path.segments.last_mut() {
        core::mem::replace(&mut last.arguments, PathArguments::None)
    } else {
        PathArguments::None
    };
    (generics, quote![#path])
}

fn map_args(args: &[Expression], context: &mut Context) -> (Vec<TokenStream>, Vec<TokenStream>) {
    let names: Vec<_> = (0..args.len()).map(|i| format_ident!("_arg_{i}")).collect();
    let values = names
        .iter()
        .zip(args.iter())
        .map(|(i, value)| {
            if matches!(value, Expression::Closure { .. }) {
                quote![]
            } else {
                let tokens = value
                    .as_const(context)
                    .unwrap_or_else(|| value.to_tokens(context));
                quote_spanned! {tokens.span()=>
                    let #i = #tokens;
                }
            }
        })
        .collect();
    let names = names
        .into_iter()
        .zip(args.iter())
        .map(|(name, value)| {
            if matches!(value, Expression::Closure { .. }) {
                value.to_tokens(context)
            } else {
                quote![#name.into()]
            }
        })
        .collect();
    (values, names)
}
