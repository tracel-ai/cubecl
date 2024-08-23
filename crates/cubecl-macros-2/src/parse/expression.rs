use quote::{format_ident, quote};
use syn::{spanned::Spanned, Expr, Lit, Type};

use crate::{
    expression::Expression,
    scope::{Context, ManagedVar},
    statement::Statement,
};

use super::{
    branch::expand_for_loop,
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
                    Expression::Verbatim {
                        tokens: quote![#path],
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
                let span = block.span();
                context.push_scope();
                let mut statements = block
                    .block
                    .stmts
                    .into_iter()
                    .map(|stmt| Statement::from_stmt(stmt, context))
                    .collect::<Result<Vec<_>, _>>()?;
                context.pop_scope();
                // Pop implicit return so we can deal with it separately instead of generating a return
                let ret = match statements.pop() {
                    Some(Statement::Expression {
                        expression,
                        terminated: false,
                        ..
                    }) => Some(expression),
                    Some(stmt) => {
                        statements.push(stmt);
                        None
                    }
                    _ => None,
                };
                let ty = ret.as_ref().and_then(|ret| ret.ty());
                Expression::Block {
                    inner: statements,
                    ret,
                    ty,
                    span,
                }
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
                let receiver = Expression::from_expr(*method.receiver, context)?;
                let args = method
                    .args
                    .into_iter()
                    .map(|arg| Expression::from_expr(arg, context))
                    .collect::<Result<Vec<_>, _>>()?;
                Expression::MethodCall {
                    receiver: Box::new(receiver),
                    method: method.method,
                    args,
                    span,
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
            Expr::Field(field) => {
                let span = field.span();
                let base = Expression::from_expr(*field.base.clone(), context)?;
                Expression::FieldAccess {
                    base: Box::new(base),
                    field: field.member,
                    span,
                }
            }
            Expr::If(_) => todo!(),
            Expr::Index(_) => todo!(),
            Expr::Infer(_) => todo!(),
            Expr::Let(_) => todo!(),
            Expr::Loop(_) => todo!(),
            Expr::Macro(_) => todo!(),
            Expr::Match(_) => todo!(),
            Expr::Paren(_) => todo!(),
            Expr::Range(_) => todo!(),
            Expr::Reference(_) => todo!(),
            Expr::Repeat(_) => todo!(),
            Expr::Return(_) => todo!(),
            Expr::Struct(_) => todo!(),
            Expr::Try(_) => todo!(),
            Expr::TryBlock(_) => todo!(),
            Expr::Tuple(_) => todo!(),
            Expr::Unsafe(_) => todo!(),
            Expr::Verbatim(_) => todo!(),
            Expr::While(_) => todo!(),
            Expr::Group(_) => todo!(),
            _ => Err(syn::Error::new_spanned(expr, "Unsupported expression"))?,
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
