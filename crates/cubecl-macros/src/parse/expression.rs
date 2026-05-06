use proc_macro2::Span;
use quote::{ToTokens, format_ident, quote, quote_spanned};
use syn::{
    Expr, ExprIndex, ExprReference, ExprUnary, Lit, LitInt, Pat, PatPath, Path, PathSegment, QSelf,
    RangeLimits, UnOp, spanned::Spanned,
};

use crate::{
    expression::{Block, Expression, MatchArm, is_intrinsic},
    generate::expression::inner_pat,
    operator::Operator,
    parse::{branch::expand_if_let, helpers::is_comptime_attr},
    scope::Context,
};

use super::{
    branch::{expand_for_loop, expand_if, expand_loop, numeric_match},
    operator::{parse_binop, parse_unop},
    statement::parse_macros,
};

// All supported primitives. Primitives don't start with an uppercase letter
pub(crate) const PRIMITIVES: &[&str] = &[
    "bool", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f16", "bf16", "f32", "f64",
    "flex32", "e2m1", "e2m1x2", "e2m3", "e3m2", "e4m3", "e5m2", "ue8m0", "usize", "isize",
];

impl Expression {
    pub fn from_expr(expr: Expr, context: &mut Context) -> syn::Result<Self> {
        let result = match expr.clone() {
            Expr::Assign(assign) => {
                let right = Self::from_expr(*assign.right, context)?;
                let left = Self::from_expr_lhs(*assign.left, context)?;
                Expression::Assignment {
                    left: Box::new(left),
                    right: Box::new(right),
                }
            }
            Expr::Binary(binary) => {
                let span = binary.span();
                let operator = parse_binop(&binary.op)?;

                let right = Self::from_expr(*binary.right, context)?;
                let left = if operator.is_assign() {
                    Self::from_expr_lhs(*binary.left, context)?
                } else {
                    Self::from_expr(*binary.left, context)?
                };

                if left.is_const() && right.is_const() {
                    let left = left.as_const(context).unwrap();
                    let right = right.as_const(context).unwrap();
                    let op = binary.op;
                    Expression::Verbatim {
                        tokens: quote![(#left #op #right)],
                    }
                } else {
                    Expression::Binary {
                        left: Box::new(left),
                        operator: parse_binop(&binary.op)?,
                        right: Box::new(right),
                        span,
                    }
                }
            }
            Expr::Lit(literal) if matches!(literal.lit, Lit::Str(_)) => Expression::Verbatim {
                tokens: literal.to_token_stream(),
            },
            Expr::Lit(literal) => Expression::Literal { value: literal.lit },
            Expr::Path(path) => {
                let name = path.path.get_ident();
                if let Some(var) = name.and_then(|ident| context.variable(ident)) {
                    if var.is_keyword {
                        Expression::Keyword { name: var.name }
                    } else {
                        Expression::Variable(var)
                    }
                } else {
                    // If it's not in the scope, it's not a managed local variable. Treat it as an
                    // external value like a Rust `const`.
                    Expression::Path {
                        path: path.path,
                        qself: path.qself,
                    }
                }
            }
            Expr::Unary(unary) => {
                let span = unary.span();
                let input = Self::from_expr(*unary.expr, context)?;
                if input.is_const() {
                    let input = input.as_const(context).unwrap();
                    let op = unary.op;
                    Expression::Verbatim {
                        tokens: quote![(#op #input)],
                    }
                } else {
                    Expression::Unary {
                        input: Box::new(input),
                        operator: parse_unop(&unary.op)?,
                        span,
                    }
                }
            }
            Expr::Block(block) => {
                let (block, _) = context.in_scope(|ctx| Block::from_block(block.block, ctx))?;
                Expression::Block(block)
            }
            Expr::Break(_) => Expression::Break,
            Expr::Call(call) => {
                let span = call.span();
                let func = Box::new(Expression::from_expr(*call.func, context)?);
                let args = call
                    .args
                    .into_iter()
                    .map(|arg| Expression::from_expr(arg, context))
                    .collect::<Result<Vec<_>, _>>()?;
                match *func {
                    Expression::Path { path, .. } if is_intrinsic(&path) => {
                        Expression::CompilerIntrinsic { func: path, args }
                    }
                    func => {
                        let associated_type = fn_associated_type(&func);
                        Expression::FunctionCall {
                            func: Box::new(func),
                            args,
                            associated_type,
                            span,
                        }
                    }
                }
            }
            Expr::MethodCall(method) => {
                let span = method.span();
                let args = method
                    .args
                    .iter()
                    .map(|arg| Expression::from_expr(arg.clone(), context))
                    .collect::<Result<Vec<_>, _>>()?;
                // Hack to at least somewhat support mutable methods
                let ends_in_mut = method.method.to_string().ends_with("_mut");
                let receiver =
                    Expression::from_expr_receiver(*method.receiver.clone(), ends_in_mut, context)?;

                if receiver.is_const()
                    && args.iter().all(|arg| arg.is_const())
                    && method.method != "runtime"
                {
                    Expression::Verbatim {
                        tokens: quote![#method],
                    }
                } else if method.method == "comptime" {
                    Expression::AssertConstant {
                        inner: Box::new(receiver),
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
                    let ty = cast.ty;
                    Expression::Verbatim {
                        tokens: quote![({#as_const} as #ty)],
                    }
                } else {
                    Expression::Cast {
                        from: Box::new(from),
                        to: *cast.ty,
                    }
                }
            }
            Expr::Const(block) => Expression::Verbatim {
                tokens: quote![#block],
            },
            Expr::Continue(cont) => Expression::Continue(cont.span()),
            Expr::Return(ret) => Expression::Return(ret.span()),
            Expr::ForLoop(for_loop) => expand_for_loop(for_loop, context)?,
            Expr::Loop(loop_expr) => expand_loop(loop_expr, context)?,
            Expr::If(if_expr) if is_let(&if_expr.cond) => expand_if_let(if_expr, context)?,
            Expr::If(if_expr) => expand_if(if_expr, context)?,
            Expr::Range(range) => {
                let span = range.span();
                let start = range
                    .start
                    .map(|start| Expression::from_expr(*start, context))
                    .transpose()?
                    .map(Box::new);
                let end = range
                    .end
                    .map(|end| Expression::from_expr(*end, context))
                    .transpose()?
                    .map(Box::new);
                Expression::Range {
                    start,
                    end,
                    span,
                    inclusive: matches!(range.limits, RangeLimits::Closed(..)),
                }
            }
            Expr::Field(field) => {
                let base = Expression::from_expr(*field.base.clone(), context)?;
                Expression::FieldAccess {
                    base: Box::new(base),
                    field: field.member,
                }
            }
            Expr::Group(group) => Expression::from_expr(*group.expr, context)?,
            Expr::Paren(paren) => Expression::from_expr(*paren.expr, context)?,
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
                let elements = tuple
                    .elems
                    .into_iter()
                    .map(|elem| Expression::from_expr(elem, context))
                    .collect::<Result<_, _>>()?;
                Expression::Tuple { elements }
            }
            Expr::Index(expr_index) => {
                let span = expr_index.span();
                let index = parse_index_expr(expr_index, context, false)?;
                if matches!(index, Expression::Verbatim { .. }) {
                    index
                } else {
                    Expression::Unary {
                        input: Box::new(index),
                        operator: Operator::Deref,
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
                let expr = Expression::from_expr(*mat.expr.clone(), context)?;
                if expr.is_const() {
                    let mut arms = Vec::new();

                    for arm in mat.arms.iter() {
                        arms.push(MatchArm {
                            pat: arm.pat.clone(),
                            expr: Box::new(Self::from_expr(arm.body.as_ref().clone(), context)?),
                        });
                    }

                    Expression::Match {
                        runtime_variants: false,
                        expr: Box::new(expr),
                        arms,
                    }
                } else if let Some(switch) = numeric_match(mat.clone(), context) {
                    switch
                } else {
                    let mut arms = Vec::new();

                    for arm in mat.arms.iter() {
                        let (expr, _) = context.in_scope(|context| {
                            add_variables_from_pat(&arm.pat, context);
                            Self::from_expr(arm.body.as_ref().clone(), context)
                        })?;
                        arms.push(MatchArm {
                            pat: arm.pat.clone(),
                            expr: Box::new(expr),
                        });
                    }

                    let runtime_compatible = arms
                        .iter()
                        .all(|arm| is_runtime_compatible_variant(&arm.pat));
                    let num_values = arms.iter().filter_map(|arm| inner_pat(&arm.pat)).count();
                    let maybe_runtime = runtime_compatible
                    && num_values <= 1
                    // Code won't work properly if there's no case. Empty or wildcard only matches
                    // are always resolved at comptime anyways.
                    && !arms.is_empty()
                    && !arms.iter().all(|arm| matches!(arm.pat, Pat::Wild(_)));
                    let is_comptime = mat.attrs.iter().any(is_comptime_attr)
                        || arms.iter().all(|it| it.expr.is_const())
                        || expr.is_const();

                    if !is_comptime && maybe_runtime {
                        let default = arms
                            .iter()
                            .find(|arm| matches!(arm.pat, Pat::Wild(_)))
                            .cloned();
                        arms.retain(|arm| !matches!(arm.pat, Pat::Wild(_)));

                        Expression::RuntimeMatch {
                            expr: Box::new(expr),
                            arms,
                            default,
                        }
                    } else {
                        Expression::Match {
                            runtime_variants: true,
                            expr: Box::new(expr),
                            arms,
                        }
                    }
                }
            }
            Expr::Macro(mac) => parse_macros(mac.mac, context)?,
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
            Expr::Unsafe(unsafe_expr) => {
                let (block, _) =
                    context.in_scope(|ctx| Block::from_block(unsafe_expr.block, ctx))?;
                Expression::Unsafe(unsafe_expr.unsafe_token, block)
            }
            Expr::Infer(_) => Expression::Verbatim { tokens: quote![_] },
            Expr::Verbatim(verbatim) => Expression::Verbatim { tokens: verbatim },
            Expr::Reference(expr_reference) => Self::from_expr_reference(expr_reference, context)?,
            Expr::Closure(expr) => {
                let (body, scope) = context.in_scope(|ctx| {
                    for arg in expr.inputs.iter() {
                        add_variables_from_pat(arg, ctx);
                    }
                    Expression::from_expr(*expr.body, ctx)
                })?;
                let body = Box::new(body);
                let params = expr.inputs.into_iter().collect();
                Expression::Closure {
                    params,
                    body,
                    scope,
                }
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

    pub fn from_expr_lhs(expr: Expr, context: &mut Context) -> syn::Result<Self> {
        let span = expr.span();
        match expr {
            // Skip deref if it's the target of an assignment
            Expr::Unary(ExprUnary {
                op: UnOp::Deref(_),
                expr,
                ..
            }) => Self::from_expr(*expr, context),
            // Skip deref and always use mutable index
            Expr::Index(ExprIndex { expr, index, .. }) => {
                let index = Self::from_expr(*index, context)?;
                let expr = Self::from_expr(*expr, context)?;
                Ok(Expression::IndexMut {
                    expr: Box::new(expr),
                    index: Box::new(index),
                    span,
                })
            }
            other => Self::from_expr(other, context),
        }
    }

    pub fn from_expr_reference(expr: ExprReference, context: &mut Context) -> syn::Result<Self> {
        let is_mutable = expr.mutability.is_some();
        let expression = match *expr.expr {
            Expr::Index(expr_index) => parse_index_expr(expr_index, context, is_mutable)?,
            Expr::Unary(ExprUnary {
                op: UnOp::Deref(_),
                expr,
                ..
            }) => {
                let span = expr.span();
                let inner = Expression::from_expr(*expr, context)?;
                let operator = match is_mutable {
                    true => Operator::AsDerefMut,
                    false => Operator::AsDeref,
                };
                Expression::Unary {
                    input: Box::new(inner),
                    operator,
                    span,
                }
            }
            other => {
                let inner = Expression::from_expr(other, context)?;
                match is_mutable {
                    true => Expression::MutReference {
                        inner: Box::new(inner),
                    },
                    false => Expression::Reference {
                        inner: Box::new(inner),
                    },
                }
            }
        };
        Ok(expression)
    }

    pub fn from_expr_receiver(
        expr: Expr,
        ends_in_mut: bool,
        context: &mut Context,
    ) -> syn::Result<Self> {
        match expr {
            // Assuming immutable is not ideal, but unsure we can fix this. For now just make atomics work.
            // Rust does a lot of magic around indexing that we can't really replicate
            Expr::Index(expr_index) => parse_index_expr(expr_index, context, ends_in_mut),
            other => Self::from_expr(other, context),
        }
    }
}

fn parse_index_expr(
    expr_index: ExprIndex,
    context: &mut Context,
    is_mut: bool,
) -> syn::Result<Expression> {
    let span = expr_index.span();
    let expr = Expression::from_expr_receiver(*expr_index.expr.clone(), is_mut, context)?;
    let index = Expression::from_expr(*expr_index.index.clone(), context)?;
    let expression = if expr.is_const() && index.is_const() {
        Expression::Verbatim {
            tokens: expr_index.to_token_stream(),
        }
    } else {
        let index = match index {
            Expression::Array { elements, span } => generate_strided_index(&expr, elements, span)?,
            index => index,
        };
        if is_mut {
            Expression::IndexMut {
                expr: Box::new(expr),
                index: Box::new(index),
                span,
            }
        } else {
            Expression::Index {
                expr: Box::new(expr),
                index: Box::new(index),
                span,
            }
        }
    };
    Ok(expression)
}

pub(crate) fn add_variables_from_pat(pat: &Pat, context: &mut Context) {
    match pat {
        Pat::Ident(pat) => {
            context.push_variable(
                pat.ident.clone(),
                None,
                false,
                pat.by_ref.is_none() && pat.mutability.is_some(),
            );
        }
        Pat::Or(pat) => pat
            .cases
            .iter()
            .for_each(|pat| add_variables_from_pat(pat, context)),
        Pat::Struct(pat) => pat.fields.iter().for_each(|f| {
            add_variables_from_pat(f.pat.as_ref(), context);
        }),
        Pat::TupleStruct(pat) => pat.elems.iter().for_each(|pat| {
            add_variables_from_pat(pat, context);
        }),
        Pat::Tuple(pat) => pat.elems.iter().for_each(|pat| {
            add_variables_from_pat(pat, context);
        }),
        _ => {}
    }
}

fn generate_strided_index(
    tensor: &Expression,
    elements: Vec<Expression>,
    span: Span,
) -> syn::Result<Expression> {
    let strided_indices = elements.into_iter().enumerate().map(|(i, elem)| {
        let i = Lit::Int(LitInt::new(&i.to_string(), span));
        let stride = Expression::MethodCall {
            receiver: Box::new(tensor.clone()),
            method: format_ident!("stride"),
            args: vec![Expression::Literal { value: i }],
            generics: None,
            span,
        };
        Expression::Binary {
            left: Box::new(elem),
            operator: Operator::Mul,
            right: Box::new(stride),
            span,
        }
    });
    let sum = strided_indices
        .reduce(|a, b| Expression::Binary {
            left: Box::new(a),
            operator: Operator::Add,
            right: Box::new(b),
            span,
        })
        .unwrap();
    Ok(sum)
}

fn fn_associated_type(path: &Expression) -> Option<(Path, Option<QSelf>, PathSegment)> {
    if !matches!(path, Expression::Path { .. }) {
        panic!("path: {path:?}");
    }

    match path {
        Expression::Path { path, qself } => {
            let second_last = path.segments.iter().nth_back(1)?;
            let name = second_last.ident.to_string();
            let ch = name.chars().next();
            let is_assoc = ch.is_some_and(|ch| ch.is_uppercase());
            let is_primitive = PRIMITIVES.contains(&name.as_str());
            if is_assoc || is_primitive {
                let mut path = path.clone();
                let name = path.segments.pop().unwrap().into_value();
                path.segments.pop_punct();
                Some((path, qself.clone(), name))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn is_let(cond: &Expr) -> bool {
    match cond {
        Expr::Group(group) => is_let(&group.expr),
        Expr::Paren(paren) => is_let(&paren.expr),
        Expr::Let(_) => true,
        _ => false,
    }
}

pub(crate) fn unwrap_noop(expr: Expr) -> Expr {
    match expr {
        Expr::Group(group) => unwrap_noop(*group.expr),
        Expr::Paren(paren) => unwrap_noop(*paren.expr),
        other => other,
    }
}

pub(crate) fn is_runtime_compatible_variant(pat: &Pat) -> bool {
    match pat {
        Pat::Ident(_) => true,
        Pat::Paren(pat) => is_runtime_compatible_variant(&pat.pat),
        Pat::Path(PatPath { .. }) => true,
        Pat::TupleStruct(pat) => pat.elems.len() == 1,
        Pat::Wild(_) => true,
        _ => false,
    }
}
