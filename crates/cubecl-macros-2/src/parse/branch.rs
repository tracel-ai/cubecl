use syn::{spanned::Spanned, Block, ExprForLoop, ExprIf, ExprLoop, ExprWhile};

use crate::{
    expression::Expression,
    scope::Context,
    statement::{parse_pat, Statement},
};

use super::helpers::Unroll;

pub fn expand_for_loop(for_loop: ExprForLoop, context: &mut Context) -> syn::Result<Expression> {
    let span = for_loop.span();
    let unroll = Unroll::from_attributes(&for_loop.attrs, context)?.map(|it| it.value);

    let right = Expression::from_expr(*for_loop.expr, context)
        .map_err(|_| syn::Error::new(span, "Unsupported for loop expression"))?;

    let (var_name, ty, _) = parse_pat(*for_loop.pat)?;
    context.push_scope();
    context.push_variable(var_name.clone(), ty.clone(), false);
    let block = parse_block(for_loop.body, context)?;
    context.pop_scope();
    Ok(Expression::ForLoop {
        range: Box::new(right),
        unroll: unroll.map(Box::new),
        var_name,
        var_ty: ty,
        block: Box::new(block),
        span,
    })
}

pub fn expand_while_loop(while_loop: ExprWhile, context: &mut Context) -> syn::Result<Expression> {
    let span = while_loop.span();

    let condition = Expression::from_expr(*while_loop.cond, context)
        .map_err(|_| syn::Error::new(span, "Unsupported while condition"))?;

    context.push_scope();
    let block = parse_block(while_loop.body, context)?;
    context.pop_scope();
    Ok(Expression::WhileLoop {
        condition: Box::new(condition),
        block: Box::new(block),
        span,
    })
}

pub fn expand_loop(loop_expr: ExprLoop, context: &mut Context) -> syn::Result<Expression> {
    let span = loop_expr.span();
    context.push_scope();
    let block = parse_block(loop_expr.body, context)?;
    context.pop_scope();
    Ok(Expression::Loop {
        block: Box::new(block),
        span,
    })
}

pub fn expand_if(if_expr: ExprIf, context: &mut Context) -> syn::Result<Expression> {
    let span = if_expr.span();
    let condition = Expression::from_expr(*if_expr.cond, context)
        .map_err(|_| syn::Error::new(span, "Unsupported while condition"))?;

    context.push_scope();
    let then_block = parse_block(if_expr.then_branch, context)?;
    context.pop_scope();
    let else_branch = if let Some((_, else_branch)) = if_expr.else_branch {
        Some(Expression::from_expr(*else_branch, context)?)
    } else {
        None
    };
    Ok(Expression::If {
        condition: Box::new(condition),
        then_block: Box::new(then_block),
        else_branch: else_branch.map(Box::new),
        span,
    })
}

pub fn parse_block(block: Block, context: &mut Context) -> syn::Result<Expression> {
    let span = block.span();

    let mut statements = block
        .stmts
        .into_iter()
        .map(|stmt| Statement::from_stmt(stmt, context))
        .collect::<Result<Vec<_>, _>>()?;
    // Pop implicit return if it exists so we can assign it as the block output
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
    Ok(Expression::Block {
        inner: statements,
        ret,
        ty,
        span,
    })
}
