use quote::{format_ident, quote};
use syn::{spanned::Spanned, Block, Expr, ExprForLoop, Meta};

use crate::{
    expression::Expression,
    scope::Context,
    statement::{parse_pat, Statement},
};

pub fn expand_for_loop(for_loop: ExprForLoop, context: &mut Context) -> syn::Result<Expression> {
    let span = for_loop.span();
    let unroll = unroll(&for_loop, context)?;

    let right = Expression::from_expr(*for_loop.expr, context)
        .map_err(|_| syn::Error::new(span, "Unsupported for loop expression"))?;

    let (var_name, ty, mutable) = parse_pat(*for_loop.pat)?;
    context.push_scope();
    context.push_variable(var_name.clone(), ty.clone(), false);
    let block = parse_block(for_loop.body, context)?;
    context.pop_scope();
    Ok(Expression::ForLoop {
        range: Box::new(right),
        unroll: unroll.map(Box::new),
        var_name,
        var_ty: ty,
        var_mut: mutable,
        block: Box::new(block),
        span,
    })
}

fn unroll(for_loop: &ExprForLoop, context: &mut Context) -> syn::Result<Option<Expression>> {
    let attribute = for_loop
        .attrs
        .iter()
        .find(|attr| {
            attr.path()
                .get_ident()
                .map(ToString::to_string)
                .map(|it| it == "unroll")
                .unwrap_or(false)
        })
        .map(|attr| match &attr.meta {
            Meta::Path(_) => quote![true],
            Meta::List(list) => list.tokens.clone(),
            Meta::NameValue(name_value) => {
                let value = &name_value.value;
                quote![#value]
            }
        });
    if let Some(attribute) = attribute {
        let expr: Expr = syn::parse2(attribute)?;
        Ok(Some(Expression::from_expr(expr, context)?))
    } else {
        Ok(None)
    }
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
