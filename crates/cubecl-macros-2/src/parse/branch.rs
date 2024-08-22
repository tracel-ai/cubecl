use quote::quote;
use syn::{spanned::Spanned, Block, ExprForLoop};

use crate::{
    expression::Expression,
    scope::Context,
    statement::{parse_pat, Statement},
};

pub fn expand_for_loop(for_loop: ExprForLoop, context: &mut Context) -> syn::Result<Expression> {
    let span = for_loop.span();
    let right = Expression::from_expr(*for_loop.expr, context)
        .map_err(|_| syn::Error::new(span, "Unsupported for loop expression"))?;
    let (from, to, step, unroll) = match right {
        Expression::FunctionCall { func, args, span } => {
            let func_name = quote![#func].to_string();
            if func_name == "range" {
                let from = args[0].clone();
                let to = args[1].clone();
                let unroll = args[2].clone();
                (from, to, None, unroll)
            } else if func_name == "range_stepped" {
                let from = args[0].clone();
                let to = args[1].clone();
                let step = args[2].clone();
                let unroll = args[3].clone();
                (from, to, Some(step), unroll)
            } else {
                Err(syn::Error::new(span, "Unsupported for loop expression"))?
            }
        }
        expr => Err(syn::Error::new(span, "Unsupported for loop expression"))?,
    };
    let (var_name, ty, mutable) = parse_pat(*for_loop.pat)?;
    context.push_scope();
    context.push_variable(var_name.clone(), ty.clone(), false);
    let statements = for_loop
        .body
        .stmts
        .into_iter()
        .map(|stmt| Statement::from_stmt(stmt, context))
        .collect::<Result<Vec<_>, _>>()?;
    context.pop_scope();
    Ok(Expression::ForLoop {
        from: Box::new(from),
        to: Box::new(to),
        step: step.map(Box::new),
        unroll: Box::new(unroll),
        var_name,
        var_ty: ty,
        var_mut: mutable,
        block: statements,
        span,
    })
}
