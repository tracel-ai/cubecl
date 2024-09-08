use std::iter;

use proc_macro2::Span;
use quote::{format_ident, quote, quote_spanned, ToTokens};
use syn::{
    parse_quote, punctuated::Punctuated, spanned::Spanned, Expr, Lit, LitInt, Path, PathSegment,
    RangeLimits, Type,
};

use crate::{
    expression::{Block, Expression},
    operator::Operator,
    scope::{Context, ManagedVar},
};

use super::{
    branch::{expand_for_loop, expand_if, expand_loop, expand_while_loop},
    operator::{parse_binop, parse_unop},
};

impl Expression {
    pub fn from_expr(expr: Expr, context: &mut Context) -> syn::Result<Self> {
        let result = match expr.clone() {
            Expr::Assign(assign) => {
                let span = assign.span();
                let right = Self::from_expr(*assign.right, context)?;
                Expression::Assigment {
                    span,
                    ty: right.ty(),
                    left: Box::new(Self::from_expr(*assign.left, context)?),
                    right: Box::new(right),
                }
            }
            Expr::Binary(binary) => {
                let span = binary.span();
                let left = Self::from_expr(*binary.left, context)?;
                let right = Self::from_expr(*binary.right, context)?;
                if left.is_const() && right.is_const() {
                    Expression::Verbatim {
                        tokens: quote![#expr],
                    }
                } else {
                    let ty = left.ty().or(right.ty());
                    Expression::Binary {
                        span,
                        left: Box::new(left),
                        operator: parse_binop(&binary.op)?,
                        right: Box::new(right),
                        ty,
                    }
                }
            }
            Expr::Lit(literal) => {
                let ty = lit_ty(&literal.lit)?;
                Expression::Literal {
                    value: literal.lit,
                    ty,
                }
            }
            Expr::Path(path) => {
                let variable = path
                    .path
                    .get_ident()
                    .and_then(|ident| context.variable(ident));
                if let Some(ManagedVar {
                    name,
                    ty,
                    is_const,
                    is_keyword,
                    ..
                }) = variable
                {
                    if is_const {
                        Expression::ConstVariable { name, ty }
                    } else if is_keyword {
                        Expression::Keyword { name }
                    } else {
                        Expression::Variable { name, ty }
                    }
                } else {
                    // If it's not in the scope, it's not a managed local variable. Treat it as an
                    // external value like a Rust `const`.
                    Expression::Path { path: path.path }
                }
            }
            Expr::Unary(unary) => {
                let span = unary.span();
                let input = Self::from_expr(*unary.expr, context)?;
                let ty = input.ty();
                Expression::Unary {
                    span,
                    input: Box::new(input),
                    operator: parse_unop(&unary.op)?,
                    ty,
                }
            }
            Expr::Block(block) => {
                let block = context.with_scope(|ctx| Block::from_block(block.block, ctx))?;
                Expression::Block(block)
            }
            Expr::Break(br) => Expression::Break { span: br.span() },
            Expr::Call(call) => {
                let span = call.span();
                let func = Box::new(Expression::from_expr(*call.func, context)?);
                let args = call
                    .args
                    .into_iter()
                    .map(|arg| Expression::from_expr(arg, context))
                    .collect::<Result<Vec<_>, _>>()?;
                let associated_type = fn_associated_type(&func);
                Expression::FunctionCall {
                    func,
                    args,
                    span,
                    associated_type,
                }
            }
            Expr::MethodCall(method) => {
                let span = method.span();
                let receiver = Expression::from_expr(*method.receiver.clone(), context)?;
                let args = method
                    .args
                    .iter()
                    .map(|arg| Expression::from_expr(arg.clone(), context))
                    .collect::<Result<Vec<_>, _>>()?;
                if receiver.is_const()
                    && args.iter().all(|arg| arg.is_const())
                    && method.method != "runtime"
                {
                    let receiver = receiver.as_const(context).unwrap();
                    let method = &method.method;
                    let args = args.iter().map(|it| it.to_tokens(context));
                    Expression::Verbatim {
                        tokens: quote![#receiver.#method(#(#args),*)],
                    }
                } else {
                    Expression::MethodCall {
                        receiver: Box::new(receiver),
                        method: method.method,
                        generics: method.turbofish,
                        args,
                        span,
                    }
                }
            }
            Expr::Cast(cast) => {
                let span = cast.span();
                let mut from_expr = *cast.expr;
                // Flatten multicasts because they shouldn't exist on the GPU
                while matches!(from_expr, Expr::Cast(_)) {
                    match from_expr {
                        Expr::Cast(cast) => from_expr = *cast.expr,
                        _ => unreachable!(),
                    }
                }
                let from = Expression::from_expr(from_expr, context)?;
                if let Some(as_const) = from.as_const(context) {
                    Expression::Verbatim { tokens: as_const }
                } else {
                    Expression::Cast {
                        from: Box::new(from),
                        to: *cast.ty,
                        span,
                    }
                }
            }
            Expr::Const(block) => Expression::Verbatim {
                tokens: quote![#block],
            },
            Expr::Continue(cont) => Expression::Continue { span: cont.span() },
            Expr::ForLoop(for_loop) => expand_for_loop(for_loop, context)?,
            Expr::While(while_loop) => expand_while_loop(while_loop, context)?,
            Expr::Loop(loop_expr) => expand_loop(loop_expr, context)?,
            Expr::If(if_expr) => expand_if(if_expr, context)?,
            Expr::Range(range) => {
                let span = range.span();
                let start = range
                    .start
                    .map(|start| Expression::from_expr(*start, context))
                    .transpose()?
                    .unwrap_or_else(|| {
                        let lit = Lit::Int(LitInt::new("0", span));
                        Expression::Literal {
                            value: lit,
                            ty: parse_quote![i32],
                        }
                    });
                let end = range
                    .end
                    .map(|end| Expression::from_expr(*end, context))
                    .transpose()?
                    .map(Box::new);
                Expression::Range {
                    start: Box::new(start),
                    end,
                    inclusive: matches!(range.limits, RangeLimits::Closed(..)),
                    span,
                }
            }
            Expr::Field(field) => {
                let span = field.span();
                let base = Expression::from_expr(*field.base.clone(), context)?;
                Expression::FieldAccess {
                    base: Box::new(base),
                    field: field.member,
                    span,
                }
            }
            Expr::Group(group) => Expression::from_expr(*group.expr, context)?,
            Expr::Paren(paren) => Expression::from_expr(*paren.expr, context)?,
            Expr::Return(ret) => Expression::Return {
                span: ret.span(),
                expr: ret
                    .expr
                    .map(|expr| Expression::from_expr(*expr, context))
                    .transpose()?
                    .map(Box::new),
                _ty: context.return_type.clone(),
            },
            Expr::Array(array) => {
                let span = array.span();
                let elements = array
                    .elems
                    .into_iter()
                    .map(|elem| Expression::from_expr(elem, context))
                    .collect::<Result<_, _>>()?;
                Expression::Array { elements, span }
            }
            Expr::Tuple(tuple) => {
                let span = tuple.span();
                let elements = tuple
                    .elems
                    .into_iter()
                    .map(|elem| Expression::from_expr(elem, context))
                    .collect::<Result<_, _>>()?;
                Expression::Tuple { elements, span }
            }
            Expr::Index(index) => {
                let span = index.span();
                let expr = Expression::from_expr(*index.expr, context)?;
                let index = Expression::from_expr(*index.index, context)?;
                if is_slice(&index) {
                    let ranges = match index {
                        Expression::Array { elements, .. } => elements.clone(),
                        Expression::Tuple { elements, .. } => elements.clone(),
                        index => vec![index],
                    };
                    Expression::Slice {
                        expr: Box::new(expr),
                        _ranges: ranges,
                        span,
                    }
                } else {
                    let index = match index {
                        Expression::Array { elements, span } => {
                            generate_strided_index(&expr, elements, span)?
                        }
                        index => index,
                    };
                    Expression::Index {
                        expr: Box::new(expr),
                        index: Box::new(index),
                        span,
                    }
                }
            }
            Expr::Repeat(repeat) => {
                let span = repeat.span();
                let len = Expression::from_expr(*repeat.len, context)?;
                if !len.is_const() {
                    Err(syn::Error::new(
                        span,
                        "Array initializer length must be known at compile time",
                    ))?
                }
                Expression::ArrayInit {
                    init: Box::new(Expression::from_expr(*repeat.expr, context)?),
                    len: Box::new(len),
                    span,
                }
            }
            Expr::Let(expr) => {
                let span = expr.span();
                let elem = Expression::from_expr(*expr.expr.clone(), context)?;
                if elem.is_const() {
                    Expression::Verbatim {
                        tokens: quote![#expr],
                    }
                } else {
                    Err(syn::Error::new(
                        span,
                        "let bindings aren't yet supported at runtime",
                    ))?
                }
            }
            Expr::Match(mat) => {
                let span = mat.span();
                let elem = Expression::from_expr(*mat.expr.clone(), context)?;
                if elem.is_const() {
                    Expression::Verbatim {
                        tokens: quote![#mat],
                    }
                } else {
                    Err(syn::Error::new(
                        span,
                        "match expressions aren't yet supported at runtime",
                    ))?
                }
            }
            Expr::Macro(mac) => Expression::Verbatim {
                tokens: quote![#mac],
            },
            Expr::Struct(init) => {
                let fields = init
                    .fields
                    .clone()
                    .into_iter()
                    .map(|field| {
                        let member = field.member;
                        let value = Expression::from_expr(field.expr, context)?;
                        syn::Result::Ok((member, value))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Expression::StructInit {
                    path: init.path,
                    fields,
                }
            }
            Expr::Unsafe(unsafe_expr) => Expression::Block(
                context.with_scope(|ctx| Block::from_block(unsafe_expr.block, ctx))?,
            ),
            Expr::Infer(_) => Expression::Verbatim { tokens: quote![_] },
            Expr::Verbatim(verbatim) => Expression::Verbatim { tokens: verbatim },
            Expr::Reference(reference) => Expression::Reference {
                inner: Box::new(Expression::from_expr(*reference.expr, context)?),
            },
            Expr::Closure(mut expr) => {
                let body = Expression::from_expr(*expr.body, context)?;
                expr.body = Box::new(Expr::Verbatim(body.to_tokens(context)));
                expr.inputs =
                    Punctuated::from_iter(iter::once(parse_quote![context]).chain(expr.inputs));
                let tokens = expr.to_token_stream();
                Expression::Closure { tokens }
            }
            Expr::Try(expr) => {
                let span = expr.span();
                let expr = Expression::from_expr(*expr.expr, context)?
                    .as_const(context)
                    .ok_or_else(|| syn::Error::new(span, "? Operator not supported at runtime"))?;
                Expression::Verbatim {
                    tokens: quote_spanned![span=> #expr?],
                }
            }
            Expr::TryBlock(_) => Err(syn::Error::new_spanned(
                expr,
                "try_blocks is unstable and not supported in kernels",
            ))?,
            e => Err(syn::Error::new_spanned(
                expr,
                format!("Unsupported expression {e:?}"),
            ))?,
        };
        Ok(result)
    }
}

fn lit_ty(lit: &Lit) -> syn::Result<Type> {
    let res = match lit {
        Lit::Int(int) => (!int.suffix().is_empty())
            .then(|| int.suffix())
            .map(|suffix| format_ident!("{suffix}"))
            .and_then(|ident| syn::parse2(quote![#ident]).ok())
            .unwrap_or_else(|| parse_quote![i32]),
        Lit::Float(float) => (!float.suffix().is_empty())
            .then(|| float.suffix())
            .map(|suffix| format_ident!("{suffix}"))
            .and_then(|ident| syn::parse2(quote![#ident]).ok())
            .unwrap_or_else(|| parse_quote![f32]),
        Lit::Bool(_) => parse_quote![bool],
        lit => Err(syn::Error::new_spanned(
            lit,
            format!("Unsupported literal type: {lit:?}"),
        ))?,
    };
    Ok(res)
}

fn generate_strided_index(
    tensor: &Expression,
    elements: Vec<Expression>,
    span: Span,
) -> syn::Result<Expression> {
    let index_ty = elements
        .first()
        .unwrap()
        .ty()
        .unwrap_or_else(|| parse_quote![u32]);
    let strided_indices = elements.into_iter().enumerate().map(|(i, elem)| {
        let i = Lit::Int(LitInt::new(&i.to_string(), span));
        let stride = Expression::MethodCall {
            receiver: Box::new(tensor.clone()),
            method: format_ident!("stride"),
            args: vec![Expression::Literal {
                value: i,
                ty: index_ty.clone(),
            }],
            span,
            generics: None,
        };
        Expression::Binary {
            left: Box::new(elem),
            operator: Operator::Mul,
            right: Box::new(stride),
            ty: None,
            span,
        }
    });
    let sum = strided_indices
        .reduce(|a, b| Expression::Binary {
            left: Box::new(a),
            operator: Operator::Add,
            right: Box::new(b),
            ty: None,
            span,
        })
        .unwrap();
    Ok(sum)
}

fn is_slice(index: &Expression) -> bool {
    match index {
        Expression::Range { .. } => true,
        Expression::Array { elements, .. } => elements.iter().any(is_slice),
        Expression::Tuple { elements, .. } => elements.iter().any(is_slice),
        _ => false,
    }
}

fn fn_associated_type(path: &Expression) -> Option<(Path, PathSegment)> {
    if !matches!(path, Expression::Path { .. }) {
        panic!("path: {path:?}");
    }
    match path {
        Expression::Path { path, .. } => {
            let is_assoc = path
                .segments
                .iter()
                .nth_back(1)
                .and_then(|it| it.ident.to_string().chars().next())
                .map(|ch| ch.is_uppercase())
                .unwrap_or(false);
            if is_assoc {
                let mut path = path.clone();
                let name = path.segments.pop().unwrap().into_value();
                path.segments.pop_punct();
                Some((path, name))
            } else {
                None
            }
        }
        _ => None,
    }
}
