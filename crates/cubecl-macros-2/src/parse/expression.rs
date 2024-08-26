use cubecl_common::operator::Operator;
use proc_macro2::Span;
use quote::{format_ident, quote, quote_spanned};
use syn::{spanned::Spanned, Expr, ExprBlock, Lit, LitInt, RangeLimits, Type};

use crate::{
    expression::Expression,
    scope::{Context, ManagedVar},
    statement::Statement,
};

use super::{
    branch::{expand_for_loop, expand_if, expand_loop, expand_while_loop, parse_block},
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
                    span: literal.span(),
                    value: literal.lit,
                    ty,
                }
            }
            Expr::Path(path) => {
                let variable = path
                    .path
                    .get_ident()
                    .and_then(|ident| context.variable(ident));
                if let Some(ManagedVar { name, ty, is_const }) = variable {
                    if is_const {
                        Expression::ConstVariable {
                            span: path.span(),
                            name,
                            ty,
                        }
                    } else {
                        Expression::Variable {
                            span: path.span(),
                            name,
                            ty,
                        }
                    }
                } else {
                    // If it's not in the scope, it's not a managed local variable. Treat it as an
                    // external value like a Rust `const`.
                    Expression::Path {
                        span: path.span(),
                        path: path.path,
                    }
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
                context.push_scope();
                let block = parse_block(block.block, context)?;
                context.pop_scope();
                block
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
                Expression::FunctionCall { func, args, span }
            }
            Expr::MethodCall(method) => {
                let span = method.span();
                let receiver = Expression::from_expr(*method.receiver.clone(), context)?;
                let args = method
                    .args
                    .iter()
                    .map(|arg| Expression::from_expr(arg.clone(), context))
                    .collect::<Result<Vec<_>, _>>()?;
                if receiver.is_const() && args.iter().all(|arg| arg.is_const()) {
                    Expression::Verbatim {
                        tokens: quote![#method],
                    }
                } else {
                    Expression::MethodCall {
                        receiver: Box::new(receiver),
                        method: method.method,
                        args,
                        span,
                    }
                }
            }
            Expr::Cast(cast) => {
                let span = cast.span();
                let from = Expression::from_expr(*cast.expr, context)?;
                Expression::Cast {
                    from: Box::new(from),
                    to: *cast.ty,
                    span,
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
                let start = *range
                    .start
                    .ok_or_else(|| syn::Error::new(span, "Open ranges not supported"))?;
                let end = *range
                    .end
                    .ok_or_else(|| syn::Error::new(span, "Open ranges not supported"))?;
                Expression::Range {
                    start: Box::new(Expression::from_expr(start, context)?),
                    end: Box::new(Expression::from_expr(end, context)?),
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
                ty: context.return_type.clone(),
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
            Expr::Index(index) => {
                let span = index.span();
                let expr = Expression::from_expr(*index.expr, context)?;
                let index = Expression::from_expr(*index.index, context)?;
                let index = match index {
                    Expression::Array { elements, span } => {
                        generate_strided_index(&expr, elements, span, context)?
                    }
                    index => index,
                };
                Expression::Index {
                    expr: Box::new(expr),
                    index: Box::new(index),
                    span,
                }
            }
            Expr::Infer(_) => todo!("infer"),
            Expr::Let(_) => todo!("let"),
            Expr::Macro(_) => todo!("macro"),
            Expr::Match(_) => todo!("match"),
            Expr::Reference(_) => todo!("reference"),
            Expr::Repeat(_) => todo!("repeat"),
            Expr::Struct(_) => todo!("struct"),
            Expr::Tuple(_) => todo!("tuple"),
            Expr::Unsafe(unsafe_expr) => {
                context.with_scope(|context| parse_block(unsafe_expr.block, context))?
            }
            Expr::Verbatim(verbatim) => Expression::Verbatim { tokens: verbatim },
            Expr::Try(_) => Err(syn::Error::new_spanned(
                expr,
                "? Operator is not supported in kernels",
            ))?,
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
            .unwrap_or_else(|| syn::parse2(quote![i32]).unwrap()),
        Lit::Float(float) => (!float.suffix().is_empty())
            .then(|| float.suffix())
            .map(|suffix| format_ident!("{suffix}"))
            .and_then(|ident| syn::parse2(quote![#ident]).ok())
            .unwrap_or_else(|| syn::parse2(quote![f32]).unwrap()),
        Lit::Bool(_) => syn::parse2(quote![bool]).unwrap(),
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
    context: &mut Context,
) -> syn::Result<Expression> {
    let index_ty = elements
        .first()
        .unwrap()
        .ty()
        .unwrap_or_else(|| syn::parse2(quote![u32]).unwrap());
    let strided_indices = elements.into_iter().enumerate().map(|(i, elem)| {
        let i = Lit::Int(LitInt::new(&i.to_string(), span));
        let stride = Expression::MethodCall {
            receiver: Box::new(tensor.clone()),
            method: format_ident!("stride"),
            args: vec![Expression::Literal {
                value: i,
                ty: index_ty.clone(),
                span,
            }],
            span,
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
