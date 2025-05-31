use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote, quote_spanned};
use syn::{GenericArgument, Ident, Member, Pat, Path, PathArguments, spanned::Spanned};

use crate::{
    expression::{Block, Expression, MatchArm},
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
                let elem = frontend_type("ExpandElementTyped");
                let frontend_path = frontend_path();
                let (array, index) = left.as_index().unwrap();
                let array = array.to_tokens(context);
                let index = index
                    .as_const(context)
                    .map(|as_const| quote![#elem::from_lit(scope, #as_const)])
                    .unwrap_or_else(|| index.to_tokens(context));
                let right = right
                    .as_const(context)
                    .map(|as_const| quote![#elem::from_lit(scope, #as_const)])
                    .unwrap_or_else(|| right.to_tokens(context));
                let op = format_ident!("{}", operator.array_op_name());
                let expand = with_span(
                    context,
                    *span,
                    quote![#frontend_path::#op::expand(scope, _array, _index.into(), _value.into())],
                );
                quote! {
                    {
                        let _array = #array;
                        let _index = #index;
                        let _value = #right;
                        #expand
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
                let expand = with_span(
                    context,
                    *span,
                    quote![#frontend_path::#op::expand(scope, _lhs.into(), _rhs.into())],
                );
                quote! {
                    {
                        let _lhs = #left;
                        let _rhs = #right;
                        #expand
                    }
                }
            }
            Expression::Unary {
                input,
                operator: Operator::Deref,
                ..
            } => input.to_tokens(context),
            Expression::Unary {
                input,
                operator,
                span,
                ..
            } => {
                let frontend_path = frontend_path();
                let input = input.to_tokens(context);
                let op = format_ident!("{}", operator.op_name());
                let expand = with_span(
                    context,
                    *span,
                    quote![#frontend_path::#op::expand(scope, _inner.into())],
                );
                quote! {
                    {
                        let _inner = #input;
                        #expand
                    }
                }
            }
            Expression::Keyword { name } => {
                quote![#name::expand(scope)]
            }
            Expression::Variable(var) => {
                if var.is_const {
                    let name = &var.name;
                    let expand_elem = frontend_type("ExpandElementTyped");
                    quote![#expand_elem::from_lit(scope, #name)]
                } else {
                    let name = &var.name;
                    if var.try_consume(context) {
                        quote![#name]
                    } else {
                        quote![#name.clone()]
                    }
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
                quote![#expand_elem::from_lit(scope, #value)]
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
                        #frontend_path::index_assign::expand(scope, _array, _index.into(), _value.into())
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
                        #frontend_path::assign::expand(scope, _value.into(), _var.into())
                    }
                }
            }
            Expression::Index { expr, index, span } => {
                let expr = expr.to_tokens(context);
                let index = index.to_tokens(context);
                let index_fn = frontend_type("index");
                let expand = with_span(
                    context,
                    *span,
                    quote![#index_fn::expand(scope, _array, _index.into())],
                );
                quote! {
                    {
                        let _array = #expr;
                        let _index = #index;
                        #expand
                    }
                }
            }
            Expression::FunctionCall {
                func,
                args,
                associated_type: None,
                span,
                ..
            } => {
                let (args, arg_names) = map_args(args, context);
                let (generics, path) = split_generics(func, context);

                let call = with_debug_call(
                    context,
                    *span,
                    quote![#path::expand #generics(scope, #(#arg_names),*)],
                );

                quote_spanned! {*span=>
                    {
                        #(#args)*
                        #call
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
                        #path::expand #generics(scope, #(#arg_names),*)
                    }
                }
            }
            Expression::FunctionCall {
                args,
                associated_type: Some((ty_path, qself, func)),
                span,
                ..
            } => {
                let ty_path = if let Some(qself) = qself {
                    let ty = &qself.ty;
                    quote![<#ty as #ty_path>]
                } else {
                    quote![#ty_path]
                };

                let (args, arg_names) = map_args(args, context);
                let mut name = func.clone();
                name.ident = format_ident!("__expand_{}", name.ident);
                let call = with_debug_call(
                    context,
                    *span,
                    quote![#ty_path::#name(scope, #(#arg_names),*)],
                );
                quote_spanned! {*span=>
                    {
                        #(#args)*
                        #call
                    }
                }
            }
            Expression::MethodCall {
                receiver,
                method,
                generics,
                args,
                span,
                ..
            } => {
                let method = format_ident!("__expand_{method}_method");
                let receiver = receiver
                    .as_const(context)
                    .unwrap_or_else(|| receiver.to_tokens(context));
                let (args, arg_names) = map_args(args, context);
                let call = with_debug_call(
                    context,
                    *span,
                    quote![#receiver.#method #generics(scope, #(#arg_names),*)],
                );
                quote_spanned! {*span=>
                    {
                        #(#args)*
                        #call
                    }
                }
            }
            Expression::Break => {
                let path = frontend_path();
                quote![#path::branch::break_expand(scope);]
            }
            Expression::Continue(span) => error!(*span, "Continue not supported yet"),
            Expression::Return(span) => error!(
                *span,
                "Return not supported yet. Consider using the terminate!() macro instead."
            ),
            Expression::Cast { from, to } => {
                let cast = prelude_type("Cast");
                let from = from.to_tokens(context);
                let to = quote_spanned![to.span()=> <#to as #cast>];
                quote! {{
                    let __from = #from;
                    #to::__expand_cast_from(scope, __from)
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
                        #for_ty::for_expand(scope, _range, _unroll, |scope, #var_name #var_ty| #block);
                    }
                }
            }
            Expression::Loop { block, scope } => {
                let loop_ty = frontend_type("branch");
                let block = context.in_fn_mut(scope, |ctx| block.to_tokens(ctx));

                quote![#loop_ty::loop_expand(scope, |scope| #block);]
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
                        #path::branch::if_else_expr_expand(scope, _cond.into(), |scope| #then_block).or_else(scope, |scope| #else_branch)
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
                        #path::branch::if_else_expand(scope, _cond.into(), |scope| #then_block).or_else(scope, |scope| #else_branch);
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
                        #path::branch::if_expand(scope, _cond.into(), |scope| #then_block);
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
                        quote![.case(scope, #val, |scope| #block)]
                    })
                    .collect::<Vec<_>>();
                quote! {
                    {
                        let _val = #value;
                        #branch::#switch(scope, _val.into(), |scope| #default)
                            #(#blocks)*
                            .finish(scope)
                    }
                }
            }
            Expression::Path { path, qself } => {
                if let Some(qself) = qself {
                    let ty = &qself.ty;
                    quote![<#ty as #path>]
                } else {
                    quote![#path]
                }
            }
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
                let turbofish = &path_last.arguments;

                let generics = match turbofish {
                    PathArguments::None => None,
                    PathArguments::AngleBracketed(params) => {
                        let params = params.args.iter().map(|p| match p {
                            GenericArgument::Type(syn::Type::Path(ty)) => {
                                if let Some(segment) = ty.path.segments.last() {
                                    GenericArgument::Type(syn::Type::Path(syn::TypePath {
                                        qself: ty.qself.clone(),
                                        path: syn::Path::from(segment.clone()),
                                    }))
                                } else {
                                    p.clone()
                                }
                            }
                            _ => p.clone(),
                        });
                        Some(quote![<#(#params),*>])
                    }
                    args => {
                        return error!(
                            args.span(),
                            "Fn generics not supported when constructing runtime structs"
                        );
                    }
                };

                let mut path_simplified = path.clone();
                if let PathArguments::AngleBracketed(params) =
                    &mut path_simplified.segments.last_mut().unwrap().arguments
                {
                    params.args.iter_mut().for_each(|p| {
                        if let GenericArgument::Type(syn::Type::Path(ty)) = p {
                            ty.path = syn::Path::from(ty.path.segments.last().unwrap().clone());
                        }
                    });
                }

                quote! {
                    {
                        type _Ty #generics = <#path_simplified as #cube_type>::ExpandType;
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
                quote![|scope, #(#params),*| #body]
            }
            Expression::Verbatim { tokens, .. } => tokens.clone(),
            Expression::Block(block) => block.to_tokens(context),
            Expression::Match {
                runtime_variants,
                expr,
                arms,
            } => {
                let arms = arms
                    .iter()
                    .map(|arm| arm.to_tokens(context, *runtime_variants));
                if *runtime_variants {
                    quote! { match (#expr).clone() { #(#arms,)* } }
                } else {
                    quote! { match #expr { #(#arms,)* } }
                }
            }
            Expression::Comment { content } => {
                let frontend_path = frontend_path();
                quote![#frontend_path::cube_comment::expand(scope, #content)]
            }
            Expression::RustMacro { ident, tokens } => {
                quote![#ident!(#tokens)]
            }
            Expression::Terminate => {
                quote![cubecl::frontend::branch::return_expand(scope);]
            }
            Expression::ExpressionMacro { ident, args } => {
                let frontend_path = frontend_path();
                let expand = format_ident!("{}_expand", ident);
                let args = args
                    .iter()
                    .map(|expr| expr.to_tokens(context))
                    .enumerate()
                    .map(|(i, arg)| {
                        let name = format_ident!("_{i}");
                        quote![let #name = #arg;]
                    });
                let arg_uses = (0..args.len()).map(|i| format_ident!("_{i}"));
                quote! {
                    {
                        #(
                            #args
                        )*
                        #frontend_path::#expand!(scope, #(#arg_uses),*);
                    }
                }
            }
        }
    }
}

impl MatchArm {
    pub fn to_tokens(&self, context: &mut Context, runtime_variants: bool) -> TokenStream {
        let mut pat = self.pat.clone();

        // If using runtime variants, we need to replace the variant Name with NameExpand.
        if runtime_variants {
            Self::expand_pat(&mut pat);
        }

        let expr = self.expr.to_tokens(context);

        quote! {
            #pat => #expr
        }
    }

    fn expand_pat(pat: &mut Pat) {
        match pat {
            // Match simple ident like x in Enum::Variant(x).
            // Useful for recursive call.
            Pat::Ident(_) => {}
            // Match path::Enum::Ident
            Pat::Path(pat) => {
                let mut path = pat.path.clone();
                append_expand_to_enum_name(&mut path);
                pat.path = path;
            }
            // Match path::Enum::Variant {a, b, c}
            Pat::Struct(pat) => {
                let mut path = pat.path.clone();
                append_expand_to_enum_name(&mut path);
                pat.path = path;
            }
            // Match path::Enum::Variant(a, b, c)
            Pat::TupleStruct(pat) => {
                let mut path = pat.path.clone();
                append_expand_to_enum_name(&mut path);
                pat.path = path;
            }
            // Match Pat1 | Pat2 | ...
            Pat::Or(pat) => {
                pat.cases.iter_mut().for_each(Self::expand_pat);
            }
            // Match (Pat1, Pat2, ...)
            Pat::Tuple(pat) => {
                pat.elems.iter_mut().for_each(Self::expand_pat);
            }
            // Match the underscore pattern _
            Pat::Wild(_) => {}
            _ => {
                panic!("unsupported pattern in match for {pat:?}");
                // NOTE: From the documentation https://docs.rs/syn/latest/syn/enum.Pat.html
                //       I don't think we should support any other patterns.
                //       Users can always use a big if, else if, else pattern instead.
                //       Currently, the goal is to support CubeType enums.
            }
        }
    }
}

// Replace something like `some_path::Enum::Variant` with `some_path::EnumExpand::Variant`.
fn append_expand_to_enum_name(path: &mut Path) {
    if path.segments.len() >= 2 {
        let segment = path.segments.get_mut(path.segments.len() - 2).unwrap(); // Safe because of the if
        segment.ident = Ident::new(&format!("{}Expand", segment.ident), Span::call_site());
    } else {
        panic!("unsupported pattern in match because of segment len");
    }
}

impl Block {
    pub fn to_tokens(&self, context: &mut Context) -> TokenStream {
        let inner: Vec<_> = self.inner.iter().map(|it| it.to_tokens(context)).collect();
        let ret = if let Some(ret) = self.ret.as_ref() {
            let as_const = ret.as_const(context);
            if let Some(as_const) = as_const {
                quote![#as_const]
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

fn init_fields<'a>(
    fields: &'a [(Member, Expression)],
    context: &'a mut Context,
) -> impl Iterator<Item = TokenStream> + 'a {
    fields.iter().map(|(pat, it)| {
        let it = if let Some(as_const) = it.as_const(context) {
            let it = quote_spanned![as_const.span()=> #as_const];
            return quote! {
                #pat: {
                    let _init = #it.clone();
                    _init.into()
                }
            };
        } else {
            it.to_tokens(context)
        };
        quote! {
            #pat: {
                let _init = #it;
                _init
            }
        }
    })
}

fn with_span(context: &Context, span: Span, tokens: TokenStream) -> TokenStream {
    if cfg!(debug_symbols) || context.debug_symbols {
        let debug_spanned = frontend_type("spanned_expand");
        quote_spanned! {span=>
            #debug_spanned(scope, line!(), column!(), |scope| #tokens)
        }
    } else {
        tokens
    }
}

fn with_debug_call(context: &Context, span: Span, tokens: TokenStream) -> TokenStream {
    if cfg!(debug_symbols) || context.debug_symbols {
        let debug_call = frontend_type("debug_call_expand");
        quote_spanned! {span=>
            #debug_call(scope, line!(), column!(), |scope| #tokens)
        }
    } else {
        tokens
    }
}
