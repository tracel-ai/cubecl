use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote, quote_spanned};
use syn::{
    GenericArgument, Ident, Member, Pat, PatIdent, PatPath, PatStruct, PatTupleStruct, Path,
    PathArguments, parse_quote, spanned::Spanned,
};

use crate::{
    expression::{Block, Expression, MatchArm},
    operator::Operator,
    paths::{frontend_path, frontend_type, prelude_path, prelude_type},
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
            } => {
                let op = format_ident!("__expand_{}_method", operator.op_name());
                let left = left.to_tokens(context);
                let right = right.to_tokens(context);
                let rhs = match operator.is_cmp() {
                    true => quote![&#right.into_expand(scope)],
                    false => quote![#right.into_expand(scope)],
                };
                let expand = with_span(
                    context,
                    *span,
                    quote![
                        #left.#op(scope, #rhs)
                    ],
                );
                quote! {{#expand}}
            }
            Expression::Unary {
                input,
                operator: Operator::Deref,
                ..
            } => {
                let input = input.to_tokens(context);
                quote! {{
                    #input.__expand_deref_method(scope)
                }}
            }
            Expression::Unary {
                input,
                operator,
                span,
                ..
            } => {
                let input = input.to_tokens(context);
                let op = format_ident!("__expand_{}_method", operator.op_name());
                let expand = with_span(context, *span, quote![#input.#op(scope)]);
                quote! {
                    {
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
                    let expand_elem = frontend_type("NativeExpand");
                    quote![#expand_elem::from_lit(scope, #name)]
                } else {
                    let name = &var.name;
                    quote![#name]
                }
            }
            Expression::FieldAccess { base, field, .. } => {
                let base = base
                    .as_const(context)
                    .unwrap_or_else(|| base.to_tokens(context));
                quote![#base.#field]
            }
            Expression::Literal { value, .. } => {
                let expand_elem = frontend_type("NativeExpand");
                quote![#expand_elem::from_lit(scope, #value)]
            }
            Expression::Assignment { left, right, .. } => {
                let right = right.to_tokens(context);
                let left = left.to_tokens(context);
                quote! {{
                    let _value = #right.into_expand(scope);
                    #left.__expand_assign_method(scope, _value)
                }}
            }
            Expression::Index { expr, index, span } => {
                let expr = expr.to_tokens(context);
                let index = index.to_tokens(context);
                let expand = with_span(
                    context,
                    *span,
                    quote![#expr.__expand_index_method(scope, #index.into_expand(scope))],
                );
                quote! {{#expand}}
            }
            Expression::IndexMut { expr, index, span } => {
                let expr = expr.to_tokens(context);
                let index = index.to_tokens(context);
                let expand = with_span(
                    context,
                    *span,
                    quote![#expr.__expand_index_mut_method(scope,  #index.into_expand(scope))],
                );
                quote! {{#expand}}
            }
            Expression::FunctionCall {
                func,
                args,
                associated_type: None,
                span,
                ..
            } => {
                let args = map_args(args, context);
                let (generics, path) = split_generics(func, context);

                let call = with_debug_call(
                    context,
                    *span,
                    quote_spanned![*span=>#path::expand #generics(scope, #(#args),*)],
                );

                quote_spanned! {*span=>{#call}}
            }
            Expression::CompilerIntrinsic { func, args } => {
                let args = map_args(args, context);
                let mut path = func.clone();
                let generics = core::mem::replace(
                    &mut path.segments.last_mut().unwrap().arguments,
                    PathArguments::None,
                );
                quote! {{
                    #path::expand #generics(scope, #(#args),*)
                }}
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

                let args = map_args(args, context);
                let mut name = func.clone();
                name.ident = format_ident!("__expand_{}", name.ident);
                let call =
                    with_debug_call(context, *span, quote![#ty_path::#name(scope, #(#args),*)]);
                quote_spanned! {*span=>{#call}}
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
                let args = map_args(args, context);
                let receiver = receiver
                    .as_const(context)
                    .unwrap_or_else(|| receiver.to_tokens(context));
                let call = with_debug_call(
                    context,
                    *span,
                    quote![#receiver.#method #generics(scope, #(#args),*)],
                );
                quote_spanned! {*span=>{#call}}
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
                    #to::__expand_cast_from(scope, #from.into_expand(scope))
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

                quote! {{
                    #for_ty::for_expand(scope, #range, #unroll, |scope, #var_name #var_ty| #block);
                }}
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
                quote! {{
                    #path::branch::if_else_expr_expand(scope, #condition.into_expand(scope), |scope| #then_block)
                        .or_else(scope, |scope| #else_branch)
                }}
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
                quote! {{
                    #path::branch::if_else_expand(scope, #condition.into_expand(scope), |scope| #then_block)
                        .or_else(scope, |scope| #else_branch);
                }}
            }
            Expression::If {
                condition,
                then_block,
                ..
            } => {
                let path = frontend_path();
                let condition = condition.to_tokens(context);
                let then_block = then_block.to_tokens(context);
                quote! {{
                    #path::branch::if_expand(scope, #condition.into_expand(scope), |scope| #then_block);
                }}
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
                        let val_tokens = val
                            .as_const(context)
                            .unwrap_or_else(|| val.to_tokens(context));
                        let block = block.to_tokens(context);
                        quote![.case(scope, #val_tokens, |scope| #block)]
                    })
                    .collect::<Vec<_>>();
                quote! {{
                    #branch::#switch(scope, #value.into_expand(scope), |scope| #default)
                        #(#blocks)*
                        .finish(scope)
                }}
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
                let start = start.as_ref().map(|start| start.to_tokens(context));
                let end = end.as_ref().map(|end| end.to_tokens(context));

                match (start, end, *inclusive) {
                    (Some(start), Some(end), false) => {
                        let range = prelude_type("RangeExpand");
                        quote! {{#range::new(#start.into(), #end.into())}}
                    }
                    (Some(start), None, false) => {
                        let range = prelude_type("RangeFromExpand");
                        quote! {{#range::new(#start.into())}}
                    }
                    (None, None, _) => {
                        let range = prelude_type("RangeFullExpand");
                        quote! {{#range{}}}
                    }
                    (Some(start), Some(end), true) => {
                        let range = prelude_type("RangeInclusiveExpand");
                        quote! {{#range::new(#start.into(), #end.into())}}
                    }
                    (None, Some(end), false) => {
                        let range = prelude_type("RangeToExpand");
                        quote! {{#range::new(#end.into())}}
                    }
                    (None, Some(end), true) => {
                        let range = prelude_type("RangeToInclusiveExpand");
                        quote! {{#range::new(#end.into())}}
                    }
                    _ => error!(*span, "Slice range not yet supported"),
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
                    quote![#inner.__expand_ref_method(scope)]
                }
            }
            Expression::MutReference { inner } => {
                if let Some(as_const) = inner.as_const(context) {
                    quote![&mut #as_const]
                } else {
                    let inner = inner.to_tokens(context);
                    quote![#inner.__expand_ref_mut_method(scope)]
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
            Expression::Unsafe(unsafe_token, block) => {
                let block = block.to_tokens(context);
                quote![#unsafe_token { #block }]
            }
            Expression::RuntimeMatch {
                expr,
                arms,
                default,
            } => {
                let branch = frontend_type("branch");

                let has_value = arms
                    .iter()
                    .next()
                    .is_some_and(|arm| arm.expr.needs_terminator());

                let match_ = match has_value {
                    true => quote![match_expand_expr],
                    false => quote![match_expand],
                };

                let expr = expr
                    .as_const(context)
                    .unwrap_or_else(|| expr.to_tokens(context));

                let arms = arms
                    .iter()
                    .map(|arm| arm.to_tokens(context, true, false))
                    .collect::<Vec<_>>();

                let default = default
                    .as_ref()
                    .map(|arm| arm.to_tokens(context, true, false))
                    .map(|(_, block)| quote![.default(scope, |scope, value| #block)]);

                let cases = arms
                    .iter()
                    .enumerate()
                    .map(|(i, (pat, block))| -> Option<_> {
                        let _ = variant_name(pat)?;
                        let discriminant = format_ident!("_disc_{i}");
                        let block = match inner_pat(pat) {
                            Some(inner) => {
                                quote! {{
                                    let #inner = value;
                                    #block
                                }}
                            }
                            None => block.clone(),
                        };
                        Some((discriminant, block))
                    })
                    .collect::<Option<Vec<_>>>();

                if let Some(mut cases) = cases {
                    let discriminants = arms.iter().enumerate().map(|(i, (pat, _))| {
                        let name = variant_name(pat).expect("Already checked");
                        let ident = format_ident!("_disc_{i}");
                        quote![let #ident = #expr.discriminant_of_value(#name);]
                    });

                    // Needed so type inference can actually work
                    let (disc0, block0) = cases.remove(0);
                    let cases = cases
                        .iter()
                        .map(|(disc, block)| quote![.case(scope, #disc, |scope, value| #block)]);
                    quote! {
                        {
                            #(#discriminants)*
                            #branch::#match_(scope, #expr, #disc0, |scope, value| #block0)
                                #(#cases)*
                                #default
                                .finish(scope)
                        }
                    }
                } else {
                    let arms = arms.iter().map(|(pat, block)| quote![#pat => #block]);

                    quote! { match #expr { #(#arms,)* } }
                }
            }
            Expression::Match {
                runtime_variants,
                expr,
                arms,
            } => {
                let is_const = self.is_const();

                let expr = expr
                    .as_const(context)
                    .unwrap_or_else(|| expr.to_tokens(context));

                let arms = arms
                    .iter()
                    .map(|arm| arm.to_tokens(context, *runtime_variants, is_const))
                    .map(|(pat, block)| quote![#pat => #block])
                    .collect::<Vec<_>>();

                quote! { match #expr { #(#arms,)* } }
            }
            Expression::RuntimeIfLet {
                expr,
                arm,
                else_branch,
            } => {
                let name = variant_name(&arm.pat);
                let if_expand = prelude_type("if_expand");
                let if_else_expand = prelude_type("if_else_expand");
                let if_else_expr_expand = prelude_type("if_else_expr_expand");

                if let Some(name) = name {
                    let expr = expr
                        .as_const(context)
                        .unwrap_or_else(|| expr.to_tokens(context));

                    let (pat, block) = arm.to_tokens(context, true, false);

                    let block = match inner_pat(&pat) {
                        Some(inner) => {
                            quote! {{
                                let #inner = #expr.runtime_value();
                                #block
                            }}
                        }
                        None => block,
                    };

                    let expand = match else_branch {
                        Some(else_branch) if else_branch.needs_terminator() => {
                            let else_branch = else_branch.to_tokens(context);
                            quote! {
                                #if_else_expr_expand(scope, __cond, |scope| #block).or_else(scope, |scope| #else_branch)
                            }
                        }
                        Some(else_branch) => {
                            let else_branch = else_branch.to_tokens(context);
                            quote! {
                                #if_else_expand(scope, __cond, |scope| #block).or_else(scope, |scope| #else_branch);
                            }
                        }
                        None => quote![#if_expand(scope, __cond, |scope| #block);],
                    };

                    quote! {{
                        let __disc = #expr.discriminant_of_value(#name).into_runtime(scope);
                        let __cond = #expr.discriminant().__expand_eq_method(scope, __disc);

                        #expand
                    }}
                } else {
                    let is_const = self.is_const();
                    let expr = expr
                        .as_const(context)
                        .unwrap_or_else(|| expr.to_tokens(context));
                    let (pat, body) = arm.to_tokens(context, true, is_const);
                    let else_branch = else_branch
                        .as_ref()
                        .map(|it| it.to_tokens(context))
                        .map(|it| quote![else #it]);
                    quote! { if let #pat = #expr #body #else_branch }
                }
            }
            Expression::IfLet {
                runtime_variants,
                expr,
                arm,
                else_branch,
            } => {
                let is_const = self.is_const();
                let expr = expr
                    .as_const(context)
                    .unwrap_or_else(|| expr.to_tokens(context));
                let (pat, body) = arm.to_tokens(context, *runtime_variants, is_const);
                let else_branch = else_branch
                    .as_ref()
                    .map(|it| it.to_tokens(context))
                    .map(|it| quote![else #it]);
                quote! { if let #pat = #expr #body #else_branch }
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
            Expression::AssertConstant { inner } => inner.to_tokens(context),
            Expression::ExpressionMacro { ident, args } => {
                let prelude = prelude_path();
                let expand = format_ident!("__expand_{}", ident);
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
                        #prelude::#expand!(scope, #(#arg_uses),*)
                    }
                }
            }
        }
    }
}

impl MatchArm {
    pub fn to_tokens(
        &self,
        context: &mut Context,
        runtime_variants: bool,
        is_const: bool,
    ) -> (Pat, TokenStream) {
        let mut pat = self.pat.clone();

        // If using runtime variants, we need to replace the variant Name with
        // NameExpand.
        if runtime_variants {
            Self::expand_pat(&mut pat);
        }

        let expr = if is_const {
            self.expr.as_const(context).unwrap()
        } else {
            self.expr.to_tokens(context)
        };

        (pat, expr)
    }

    fn expand_pat(pat: &mut Pat) {
        match pat {
            Pat::Ident(ident) if ident.ident == "None" => {
                *pat = Pat::Path(parse_quote![OptionExpand::None]);
            }
            Pat::Ident(PatIdent {
                subpat: Some((_, pat)),
                ..
            }) => {
                Self::expand_pat(pat);
            }
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
                //       Users can always use a big if, else if, else pattern
                // instead.       Currently, the goal is to
                // support CubeType enums.
            }
        }
    }
}

// Replace something like `some_path::Enum::Variant` with
// `some_path::EnumExpand::Variant`.
fn append_expand_to_enum_name(path: &mut Path) {
    if path.segments.len() >= 2 {
        let segment = path.segments.get_mut(path.segments.len() - 2).unwrap(); // Safe because of the if
        segment.ident = Ident::new(&format!("{}Expand", segment.ident), Span::call_site());
    } else if path.is_ident("Some") || path.is_ident("None") {
        // Insert CubeOption for prelude use of `Some` and `None`
        *path = parse_quote!(OptionExpand::#path);
    } else {
        panic!("Found single path {path:?}");
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

fn map_args(args: &[Expression], context: &mut Context) -> Vec<TokenStream> {
    args.iter()
        .map(|value| {
            let is_closure = is_closure(value);
            let tokens = value
                .as_const(context)
                .unwrap_or_else(|| value.to_tokens(context));
            if is_closure {
                tokens
            } else {
                quote_spanned![tokens.span()=> (#tokens).into()]
            }
        })
        .collect()
}

fn is_closure(expr: &Expression) -> bool {
    match expr {
        Expression::Reference { inner } | Expression::MutReference { inner } => is_closure(inner),
        Expression::Closure { .. } => true,
        _ => false,
    }
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
                    #it.into()
                }
            };
        } else {
            it.to_tokens(context)
        };
        quote! {
            #pat: {
                #it
            }
        }
    })
}

fn with_span(context: &Context, span: Span, tokens: TokenStream) -> TokenStream {
    if context.debug_symbols {
        quote_spanned! {span=>
            scope.update_span(line!(), column!());
            #tokens
        }
    } else {
        tokens
    }
}

fn with_debug_call(context: &Context, span: Span, tokens: TokenStream) -> TokenStream {
    if context.debug_symbols {
        let debug_call = frontend_type("debug_call_expand");
        quote_spanned! {span=>
            #debug_call(scope, line!(), column!(), |scope| #tokens)
        }
    } else {
        tokens
    }
}

fn variant_name(pat: &Pat) -> Option<String> {
    match pat {
        Pat::Ident(pat) => Some(pat.ident.to_string()),
        Pat::Paren(pat) => variant_name(&pat.pat),
        Pat::Path(PatPath { path, .. })
        | Pat::Struct(PatStruct { path, .. })
        | Pat::TupleStruct(PatTupleStruct { path, .. }) => {
            path.segments.last().map(|it| it.ident.to_string())
        }
        _ => None,
    }
}

pub(crate) fn inner_pat(pat: &Pat) -> Option<Pat> {
    match pat {
        Pat::Ident(_) => None,
        Pat::Paren(pat) => inner_pat(&pat.pat),
        Pat::Path(PatPath { .. }) => None,
        Pat::TupleStruct(pat) => pat.elems.first().cloned(),
        _ => None,
    }
}
