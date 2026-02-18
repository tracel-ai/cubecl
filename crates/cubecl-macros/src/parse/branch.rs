use quote::quote;
use syn::{
    Expr, ExprForLoop, ExprIf, ExprLoop, ExprMatch, Ident, Pat, parse_quote, spanned::Spanned,
};

use crate::{
    expression::{Block, Expression, MatchArm},
    parse::expression::{add_variables_from_pat, unwrap_noop},
    scope::Context,
    statement::Statement,
};

use super::{helpers::Unroll, statement::parse_pat};

pub fn expand_for_loop(
    mut for_loop: ExprForLoop,
    context: &mut Context,
) -> syn::Result<Expression> {
    let span = for_loop.span();
    let unroll = Unroll::from_attributes(&for_loop.attrs, context)?;
    let var = parse_pat(*for_loop.pat)?;

    if let Some(Unroll {
        always_true: true, ..
    }) = unroll
        && var.ident != "_"
    {
        let var_name = &var.ident;
        for_loop.body.stmts.insert(
            0,
            parse_quote![let #var_name = #var_name.into_lit_unchecked().comptime();],
        );
    };

    let unroll = unroll.map(|it| it.value);
    let right = Expression::from_expr(*for_loop.expr.clone(), context)
        .map_err(|_| syn::Error::new(span, "Unsupported for loop expression"))?;

    if right.is_const() && !matches!(right, Expression::Range { .. }) {
        return expand_for_in_loop(var.ident, right, for_loop.body, context);
    }

    let (block, scope) = context.in_scope(|context| {
        context.push_variable(
            var.ident.clone(),
            var.ty.clone(),
            false,
            var.is_ref,
            var.is_mut,
        );
        Block::from_block(for_loop.body, context)
    })?;

    Ok(Expression::ForLoop {
        range: Box::new(right),
        unroll: unroll.map(Box::new),
        var_name: var.ident,
        var_ty: var.ty,
        block,
        scope,
    })
}

fn expand_for_in_loop(
    var_name: Ident,
    right: Expression,
    block: syn::Block,
    context: &mut Context,
) -> syn::Result<Expression> {
    let statements = block
        .stmts
        .into_iter()
        .map(|stmt| Statement::from_stmt(stmt, context))
        .collect::<Result<Vec<_>, _>>()?;

    let right = right.to_tokens(context);
    let statements = statements.into_iter().map(|it| it.to_tokens(context));
    let for_loop = Expression::VerbatimTerminated {
        tokens: quote! {
            for #var_name in #right {
                #(#statements)*
            }
        },
    };
    Ok(for_loop)
}

pub fn expand_loop(loop_expr: ExprLoop, context: &mut Context) -> syn::Result<Expression> {
    let (block, scope) = context.in_scope(|ctx| Block::from_block(loop_expr.body, ctx))?;
    Ok(Expression::Loop { block, scope })
}

pub fn expand_if(if_expr: ExprIf, context: &mut Context) -> syn::Result<Expression> {
    let span = if_expr.span();
    let condition = Expression::from_expr(*if_expr.cond, context)
        .map_err(|_| syn::Error::new(span, "Unsupported if condition"))?;

    let (then_block, _) = context.in_scope(|ctx| Block::from_block(if_expr.then_branch, ctx))?;
    let else_branch = if let Some((_, else_branch)) = if_expr.else_branch {
        let (expr, _) = context.in_scope(|ctx| Expression::from_expr(*else_branch, ctx))?;
        Some(Box::new(expr))
    } else {
        None
    };
    Ok(Expression::If {
        condition: Box::new(condition),
        then_block,
        else_branch,
    })
}

pub fn expand_if_let(if_expr: ExprIf, context: &mut Context) -> syn::Result<Expression> {
    let Expr::Let(let_expr) = unwrap_noop(*if_expr.cond.clone()) else {
        unreachable!()
    };

    let expr = Expression::from_expr(*let_expr.expr.clone(), context)?;
    let runtime_variants = !expr.is_const();

    let (then_block, _) = context.in_scope(|ctx| {
        if !expr.is_const() {
            add_variables_from_pat(&let_expr.pat, ctx);
        }
        Block::from_block(if_expr.then_branch, ctx)
    })?;
    let else_branch = if let Some((_, else_branch)) = if_expr.else_branch {
        let (expr, _) = context.in_scope(|ctx| Expression::from_expr(*else_branch, ctx))?;
        Some(Box::new(expr))
    } else {
        None
    };

    let arm = MatchArm {
        pat: *let_expr.pat,
        expr: Box::new(Expression::Block(then_block)),
    };

    Ok(Expression::IfLet {
        runtime_variants,
        expr: Box::new(expr),
        arm,
        else_branch,
    })
}

pub fn numeric_match(mat: ExprMatch, context: &mut Context) -> Option<Expression> {
    fn parse_pat(pat: Pat) -> Option<Vec<Expression>> {
        match pat {
            Pat::Lit(lit) => Some(vec![Expression::Literal { value: lit.lit }]),
            Pat::Or(or) => {
                let pats = or
                    .cases
                    .into_iter()
                    .map(parse_pat)
                    .collect::<Option<Vec<_>>>()?;
                Some(pats.into_iter().flatten().collect())
            }
            Pat::Wild(_) => Some(vec![]),
            Pat::Path(pat) => Some(vec![Expression::Path {
                path: pat.path,
                qself: pat.qself,
            }]),
            Pat::Ident(pat) => Some(vec![Expression::Path {
                path: syn::Path::from(pat.ident),
                qself: None,
            }]),
            _ => None,
        }
    }

    fn parse_body(expr: Expr, context: &mut Context) -> Option<Block> {
        match expr {
            Expr::Block(block) => Block::from_block(block.block, context).ok(),
            expr => {
                let expr = Expression::from_expr(expr, context).ok()?;
                Some(Block {
                    ret: Some(Box::new(expr)),
                    inner: vec![],
                })
            }
        }
    }

    let value = Box::new(Expression::from_expr(*mat.expr, context).ok()?);

    let arms = mat
        .arms
        .into_iter()
        .map(|arm| arm.guard.is_none().then_some(arm))
        .collect::<Option<Vec<_>>>()?;

    let mut switch_arms = Vec::new();
    let mut default = None;

    for arm in arms {
        let pat = parse_pat(arm.pat)?;
        if pat.is_empty() {
            default = Some(arm.body)
        } else {
            switch_arms.extend(pat.into_iter().map(|lit| (lit, arm.body.clone())));
        }
    }

    let default = parse_body(*default?, context)?;

    let cases = switch_arms
        .into_iter()
        .map(|(lit, body)| Some((lit, parse_body(*body, context)?)))
        .collect::<Option<Vec<_>>>()?;

    Some(Expression::Switch {
        value,
        cases,
        default,
    })
}

impl Block {
    pub fn from_block(block: syn::Block, context: &mut Context) -> syn::Result<Self> {
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
        Ok(Self {
            inner: statements,
            ret,
        })
    }
}
