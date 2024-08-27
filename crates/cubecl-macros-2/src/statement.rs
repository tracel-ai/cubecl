use crate::{expression::Expression, scope::Context};
use proc_macro2::Span;
use syn::{spanned::Spanned, Ident, Pat, PatStruct, Stmt, Type};

#[derive(Clone, Debug)]
pub enum Statement {
    Local {
        left: Box<Expression>,
        init: Option<Box<Expression>>,
        mutable: bool,
        ty: Option<Type>,
        span: Span,
    },
    Destructure {
        fields: Vec<(Pat, Expression)>,
        span: Span,
    },
    Expression {
        expression: Box<Expression>,
        terminated: bool,
        span: Span,
    },
}

impl Statement {
    pub fn from_stmt(stmt: Stmt, context: &mut Context) -> syn::Result<Self> {
        let statement = match stmt {
            Stmt::Local(local) => {
                let span = local.span();

                let init = local
                    .init
                    .map(|init| Expression::from_expr(*init.expr, context))
                    .transpose()?
                    .map(Box::new);
                let (ident, ty, mutable) = match local.pat {
                    Pat::Struct(pat) => {
                        return parse_struct_destructure(pat, *init.unwrap(), context);
                    }
                    pat => parse_pat(pat)?,
                };
                let is_const = init.as_ref().map(|init| init.is_const()).unwrap_or(false);
                let variable = Box::new(Expression::Variable {
                    name: ident.clone(),
                    span,
                    ty: ty.clone(),
                });

                context.push_variable(ident, ty.clone(), is_const && !mutable);
                Self::Local {
                    left: variable,
                    init,
                    mutable,
                    ty,
                    span,
                }
            }
            Stmt::Expr(expr, semi) => {
                let span = expr.span();
                let expression = Box::new(Expression::from_expr(expr, context)?);
                Statement::Expression {
                    terminated: semi.is_some() || !expression.needs_terminator(),
                    span,
                    expression,
                }
            }
            stmt => Err(syn::Error::new_spanned(stmt, "Unsupported statement"))?,
        };
        Ok(statement)
    }
}

pub fn parse_pat(pat: Pat) -> syn::Result<(Ident, Option<Type>, bool)> {
    let res = match pat {
        Pat::Ident(ident) => (ident.ident, None, ident.mutability.is_some()),
        Pat::Type(pat) => {
            let ty = *pat.ty;
            let (ident, _, mutable) = parse_pat(*pat.pat)?;
            (ident, Some(ty), mutable)
        }
        pat => Err(syn::Error::new_spanned(
            pat.clone(),
            format!("Unsupported local pat: {pat:?}"),
        ))?,
    };
    Ok(res)
}

fn parse_struct_destructure(
    pat: PatStruct,
    init: Expression,
    context: &mut Context,
) -> syn::Result<Statement> {
    let fields = pat
        .fields
        .into_iter()
        .map(|field| {
            let span = field.span();
            let access = Expression::FieldAccess {
                base: Box::new(init.clone()),
                field: field.member,
                span,
            };
            let (ident, ty, _) = parse_pat(*field.pat.clone())?;
            context.push_variable(ident.clone(), ty.clone(), init.is_const());
            Ok((*field.pat, access))
        })
        .collect::<syn::Result<Vec<_>>>()?;

    Ok(Statement::Destructure {
        fields,
        span: Span::call_site(),
    })
}
