use proc_macro2::TokenStream;
use quote::{format_ident, quote, quote_spanned};
use syn::{spanned::Spanned, Member, PathArguments};

use crate::{
    expression::{Block, ConstMatchArm, Expression},
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
                ..
            } if operator.is_assign() && matches!(**left, Expression::Index { .. }) => {
                let elem = frontend_type("ExpandElementTyped");
                let frontend_path = frontend_path();
                let (array, index) = left.as_index().unwrap();
                let array = array.to_tokens(context);
                let index = index
                    .as_const(context)
                    .map(|as_const| quote![#elem::from_lit(#as_const)])
                    .unwrap_or_else(|| index.to_tokens(context));
                let right = right
                    .as_const(context)
                    .map(|as_const| quote![#elem::from_lit(#as_const)])
                    .unwrap_or_else(|| right.to_tokens(context));
                let op = format_ident!("{}", operator.array_op_name());
                quote! {
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
                ..
            } => {
                let frontend_path = frontend_path();
                let op = format_ident!("{}", operator.op_name());
                let left = left.to_tokens(context);
                let right = right.to_tokens(context);
                quote! {
                    {
                        let _lhs = #left;
                        let _rhs = #right;
                        #frontend_path::#op::expand(context, _lhs, _rhs)
                    }
                }
            }
            Expression::Unary {
                input,
                operator: Operator::Deref,
                ..
            } => input.to_tokens(context),
            Expression::Unary {
                input, operator, ..
            } => {
                let frontend_path = frontend_path();
                let input = input.to_tokens(context);
                let op = format_ident!("{}", operator.op_name());
                quote! {
                    {
                        let _inner = #input;
                        #frontend_path::#op::expand(context, _inner)
                    }
                }
            }
            Expression::Keyword { name } => {
                quote![#name::expand(context)]
            }
            Expression::Variable(var) if var.is_const => {
                let name = &var.name;
                let expand_elem = frontend_type("ExpandElementTyped");
                quote![#expand_elem::from_lit(#name)]
            }
            Expression::Variable(var) => {
                let name = &var.name;
                if var.try_consume(context) {
                    quote![#name]
                } else {
                    quote![#name.clone()]
                }
            }

            Expression::FieldAccess { base, field, .. } => {
                let base = base
                    .as_const(context)
                    .unwrap_or_else(|| base.to_tokens(context));
                quote![#base.#field.clone()]
            }
            Expression::Literal { value, .. } => {
                let expand_elem = frontend_type("ExpandElementTyped");
                quote![#expand_elem::from_lit(#value)]
            }

            Expression::Assignment { left, right, .. }
                if matches!(**left, Expression::Index { .. }) =>
            {
                let (array, index) = left.as_index().unwrap();
                let array = array.to_tokens(context);
                let index = index.to_tokens(context);
                let right = right.to_tokens(context);
                let frontend_path = frontend_path();
                quote! {
                    {
                        let _array = #array;
                        let _index = #index;
                        let _value = #right;
                        #frontend_path::index_assign::expand(context, _array, _index, _value)
                    }
                }
            }
            Expression::Assignment { left, right, .. } => {
                let frontend_path = frontend_path();
                let left = left.to_tokens(context);
                let right = right.to_tokens(context);
                quote! {
                    {
                        let _var = #left;
                        let _value = #right;
                        #frontend_path::assign::expand(context, _value, _var)
                    }
                }
            }
            Expression::Index { expr, index } => {
                let expr = expr.to_tokens(context);
                let index = index.to_tokens(context);
                let index_fn = frontend_type("index");
                quote! {
                    {
                        let _array = #expr;
                        let _index = #index;
                        #index_fn::expand(context, _array, _index)
                    }
                }
            }
            Expression::FunctionCall {
                func,
                args,
                associated_type: None,
                ..
            } => {
                let (args, arg_names) = map_args(args, context);
                let (generics, path) = split_generics(func, context);
                quote! {
                    {
                        #(#args)*
                        #path::expand #generics(context, #(#arg_names),*)
                    }
                }
            }
            Expression::CompilerIntrinsic { func, args } => {
                let (args, arg_names) = map_args(args, context);
                let mut path = func.clone();
                let generics = core::mem::replace(
                    &mut path.segments.last_mut().unwrap().arguments,
                    PathArguments::None,
                );
                quote! {
                    {
                        #(#args)*
                        #path::expand #generics(context, #(#arg_names),*)
                    }
                }
            }
            Expression::FunctionCall {
                args,
                associated_type: Some((ty_path, func)),
                ..
            } => {
                let (args, arg_names) = map_args(args, context);
                let mut name = func.clone();
                name.ident = format_ident!("__expand_{}", name.ident);
                quote! {
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
                ..
            } => {
                let method = format_ident!("__expand_{method}_method");
                let receiver = receiver
                    .as_const(context)
                    .unwrap_or_else(|| receiver.to_tokens(context));
                let (args, arg_names) = map_args(args, context);
                quote! {
                    {
                        #(#args)*
                        #receiver.#method #generics(context, #(#arg_names),*)
                    }
                }
            }
            Expression::Break => {
                let path = frontend_path();
                quote![#path::branch::break_expand(context);]
            }
            Expression::Continue(span) => error!(*span, "Continue not supported yet"),
            Expression::Return { expr, span, .. } => {
                if expr.is_some() {
                    error!(*span, "Only void return is supported.")
                } else {
                    quote![cubecl::frontend::branch::return_expand(context);]
                }
            }
            Expression::Cast { from, to } => {
                let cast = prelude_type("Cast");
                let from = from.to_tokens(context);
                let to = quote_spanned![to.span()=> <#to as #cast>];
                quote! {{
                    let __from = #from;
                    #to::__expand_cast_from(context, __from)
                }}
            }
            Expression::ForLoop {
                range,
                unroll,
                var_name,
                var_ty,
                block,
                scope,
            } => {
                let for_ty = frontend_type("branch");

                let range = range.to_tokens(context);
                let unroll = unroll
                    .as_ref()
                    .and_then(|it| it.as_const(context))
                    .unwrap_or(quote![false]);
                let block = context.in_fn_mut(scope, |ctx| block.to_tokens(ctx));
                let var_ty = var_ty.as_ref().map(|it| quote![: #it]);

                quote! {
                    {
                        let _range = #range;
                        let _unroll = #unroll;
                        #for_ty::for_expand(context, _range, _unroll, |context, #var_name #var_ty| #block);
                    }
                }
            }
            Expression::Loop { block, scope } => {
                let loop_ty = frontend_type("branch");
                let block = context.in_fn_mut(scope, |ctx| block.to_tokens(ctx));

                quote![#loop_ty::loop_expand(context, |context| #block);]
            }
            Expression::If {
                condition,
                then_block,
                else_branch,
            } if condition.is_const() => {
                let as_const = condition.as_const(context).unwrap();
                let then_block = then_block.to_tokens(context);
                let else_branch = else_branch
                    .as_ref()
                    .map(|it| it.to_tokens(context))
                    .map(|it| quote![else #it]);
                quote![if #as_const #then_block #else_branch]
            }
            Expression::If {
                condition,
                then_block,
                else_branch: Some(else_branch),
            } if then_block.ret.is_some() && else_branch.needs_terminator() => {
                let path = frontend_path();
                let condition = condition.to_tokens(context);
                let then_block = then_block.to_tokens(context);
                let else_branch = else_branch.to_tokens(context);
                quote! {
                    {
                        let _cond = #condition;
                        #path::branch::if_else_expr_expand(context, _cond.into(), |context| #then_block).or_else(context, |context| #else_branch)
                    }
                }
            }
            Expression::If {
                condition,
                then_block,
                else_branch: Some(else_branch),
            } => {
                let path = frontend_path();
                let condition = condition.to_tokens(context);
                let then_block = then_block.to_tokens(context);
                let else_branch = else_branch.to_tokens(context);
                quote! {
                    {
                        let _cond = #condition;
                        #path::branch::if_else_expand(context, _cond.into(), |context| #then_block).or_else(context, |context| #else_branch);
                    }
                }
            }
            Expression::If {
                condition,
                then_block,
                ..
            } => {
                let path = frontend_path();
                let condition = condition.to_tokens(context);
                let then_block = then_block.to_tokens(context);
                quote! {
                    {
                        let _cond = #condition;
                        #path::branch::if_expand(context, _cond.into(), |context| #then_block);
                    }
                }
            }
            Expression::Switch {
                value,
                cases,
                default,
            } => {
                let branch = frontend_type("branch");
                let switch = match default.ret.is_some() {
                    true => quote![switch_expand_expr],
                    false => quote![switch_expand],
                };
                let value = value.to_tokens(context);
                let default = default.to_tokens(context);
                let blocks = cases
                    .iter()
                    .map(|(val, block)| {
                        let block = block.to_tokens(context);
                        quote![.case(context, #val, |context| #block)]
                    })
                    .collect::<Vec<_>>();
                quote! {
                    {
                        let _val = #value;
                        #branch::#switch(context, _val.into(), |context| #default)
                            #(#blocks)*
                            .finish(context)
                    }
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
                    let range = frontend_type("RangeExpand");
                    let end = end
                        .as_const(context)
                        .unwrap_or_else(|| end.to_tokens(context));
                    quote! {
                        {
                            let _start = #start;
                            let _end = #end;
                            #range::new(_start.into(), _end.into(), #inclusive)
                        }
                    }
                } else {
                    error!(*span, "Slice range not yet supported")
                }
            }

            Expression::Array { span, .. } => {
                if let Some(constant) = self.as_const(context) {
                    constant
                } else {
                    error!(*span, "Array expressions can't be used at runtime")
                }
            }
            Expression::Tuple { elements, .. } => {
                if let Some(constant) = self.as_const(context) {
                    constant
                } else {
                    let elements = elements.iter().map(|it| it.to_tokens(context));
                    quote![(#(#elements),*)]
                }
            }

            Expression::Slice { span, .. } => {
                error!(*span, "Slice expressions not yet implemented")
            }
            Expression::ArrayInit { init, len } => {
                let init_ty = frontend_type("ArrayInit");
                let init = init.to_tokens(context);
                let len = len.to_tokens(context);

                quote![#init_ty::new(#len, #init)]
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
                let fields = init_fields(fields, context);
                let path_last = path.segments.last().unwrap();
                let turbofish = path_last.arguments.clone();
                let generics = match &turbofish {
                    PathArguments::None => None,
                    PathArguments::AngleBracketed(params) => {
                        let params = params.args.iter();
                        Some(quote![<#(#params),*>])
                    }
                    args => {
                        return error!(
                            args.span(),
                            "Fn generics not supported when constructing runtime structs"
                        )
                    }
                };

                quote! {
                    {
                        type _Ty #generics = <#path as #cube_type>::ExpandType;
                        _Ty #turbofish { #(#fields),* }
                    }
                }
            }
            Expression::Closure {
                params,
                body,
                scope,
            } => {
                // Without knowing the closure type, we need to assume it's `FnMut`
                let body = context.in_fn_mut(scope, |ctx| body.to_tokens(ctx));
                quote![|context, #(#params),*| #body]
            }
            Expression::Verbatim { tokens, .. } => tokens.clone(),
            Expression::Block(block) => block.to_tokens(context),
            Expression::ConstMatch { const_expr, arms } => {
                let arms = arms.iter().map(|arm| arm.to_tokens(context));

                quote! {
                    match #const_expr {
                        #(#arms,)*
                    }
                }
            }
        }
    }
}

impl ConstMatchArm {
    pub fn to_tokens(&self, context: &mut Context) -> TokenStream {
        let path = &self.pat;
        let expr = self.expr.to_tokens(context);

        quote! {
            #path => #expr
        }
    }
}

impl Block {
    pub fn to_tokens(&self, context: &mut Context) -> TokenStream {
        let inner: Vec<_> = self.inner.iter().map(|it| it.to_tokens(context)).collect();
        let ret = if let Some(ret) = self.ret.as_ref() {
            let as_const = ret.as_const(context);
            if let Some(as_const) = as_const {
                quote![#as_const.__expand_runtime_method(context)]
            } else {
                ret.to_tokens(context)
            }
        } else {
            quote![()]
        };

        quote! {
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
                quote_spanned![tokens.span()=> let #i = #tokens;]
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

/// Since we no longer (unnecessarily) init immutable locals, we do need to init all struct fields
/// because of interior mutability.
fn init_fields<'a>(
    fields: &'a [(Member, Expression)],
    context: &'a mut Context,
) -> impl Iterator<Item = TokenStream> + 'a {
    fields.iter().map(|(pat, it)| {
        let init = frontend_type("Init");
        let it = if let Some(as_const) = it.as_const(context) {
            let expand_elem = frontend_type("ExpandElementTyped");
            quote_spanned![as_const.span()=> #expand_elem::from_lit(#as_const)]
        } else {
            it.to_tokens(context)
        };
        quote! {
            #pat: {
                let _init = #it;
                #init::init(_init, context)
            }
        }
    })
}
