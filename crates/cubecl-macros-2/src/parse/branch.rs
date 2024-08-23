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
    let statements = for_loop
        .body
        .stmts
        .into_iter()
        .map(|stmt| Statement::from_stmt(stmt, context))
        .collect::<Result<Vec<_>, _>>()?;
    context.pop_scope();
    Ok(Expression::ForLoop {
        range: Box::new(right),
        unroll: Box::new(unroll),
        var_name,
        var_ty: ty,
        var_mut: mutable,
        block: statements,
        span,
    })
}

fn unroll(for_loop: &ExprForLoop, context: &mut Context) -> syn::Result<Expression> {
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
    let attribute = attribute.unwrap_or_else(|| quote![false]);
    let expr: Expr = syn::parse2(attribute)?;
    Expression::from_expr(expr, context)
}
