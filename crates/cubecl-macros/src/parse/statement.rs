use quote::format_ident;
use syn::{
    spanned::Spanned, Ident, Index, Member, Pat, PatStruct, PatTuple, PatTupleStruct, Stmt, Type,
    TypeReference,
};

use crate::{
    expression::Expression,
    scope::Context,
    statement::{Pattern, Statement},
};

impl Statement {
    pub fn from_stmt(stmt: Stmt, context: &mut Context) -> syn::Result<Self> {
        let statement = match stmt {
            Stmt::Local(local) => {
                let init = local
                    .init
                    .map(|init| Expression::from_expr(*init.expr, context))
                    .transpose()?
                    .map(Box::new);
                let Pattern {
                    ident,
                    ty,
                    is_ref,
                    is_mut,
                } = match local.pat {
                    Pat::Struct(pat) => {
                        return desugar_struct_local(pat, *init.unwrap(), context);
                    }
                    Pat::Tuple(PatTuple { elems, .. })
                    | Pat::TupleStruct(PatTupleStruct { elems, .. }) => {
                        return desugar_tuple_local(elems, *init.unwrap(), context)
                    }
                    pat => parse_pat(pat)?,
                };
                let is_const = init.as_ref().map(|init| init.is_const()).unwrap_or(false);

                let variable =
                    context.push_variable(ident, ty, is_const && !is_mut, is_ref, is_mut);
                Self::Local { variable, init }
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
            Stmt::Item(_) => Statement::Skip,
            stmt => Err(syn::Error::new_spanned(stmt, "Unsupported statement"))?,
        };
        Ok(statement)
    }
}

pub fn parse_pat(pat: Pat) -> syn::Result<Pattern> {
    let res = match pat {
        Pat::Ident(ident) => Pattern {
            ident: ident.ident,
            ty: None,
            is_ref: ident.by_ref.is_some(),
            is_mut: ident.mutability.is_some(),
        },
        Pat::Type(pat) => {
            let ty = *pat.ty;
            let is_ref = matches!(ty, Type::Reference(_));
            let ref_mut = matches!(
                ty,
                Type::Reference(TypeReference {
                    mutability: Some(_),
                    ..
                })
            );
            let inner = parse_pat(*pat.pat)?;
            Pattern {
                ident: inner.ident,
                ty: Some(ty),
                is_ref: is_ref || inner.is_ref,
                is_mut: ref_mut || inner.is_mut,
            }
        }
        Pat::Wild(_) => Pattern {
            ident: format_ident!("_"),
            ty: None,
            is_ref: false,
            is_mut: false,
        },
        pat => Err(syn::Error::new_spanned(
            pat.clone(),
            format!("Unsupported local pat: {pat:?}"),
        ))?,
    };
    Ok(res)
}

fn desugar_struct_local(
    pat: PatStruct,
    init: Expression,
    context: &mut Context,
) -> syn::Result<Statement> {
    let temp_name = format_ident!("__struct_destructure_init");
    let temp = create_local(temp_name.clone(), init, context);
    let mut fields = pat
        .fields
        .into_iter()
        .map(|field| desugar_field(field.member, *field.pat, &temp_name, context))
        .collect::<syn::Result<Vec<_>>>()?;
    fields.insert(0, temp);

    Ok(Statement::Group { statements: fields })
}

fn desugar_tuple_local(
    elems: impl IntoIterator<Item = Pat>,
    init: Expression,
    context: &mut Context,
) -> syn::Result<Statement> {
    let temp_name = format_ident!("__tuple_destructure_init");
    let temp = create_local(temp_name.clone(), init, context);
    let mut fields = elems
        .into_iter()
        .enumerate()
        .map(|(i, pat)| {
            let member = Member::Unnamed(Index::from(i));
            desugar_field(member, pat, &temp_name, context)
        })
        .collect::<syn::Result<Vec<_>>>()?;
    fields.insert(0, temp);

    Ok(Statement::Group { statements: fields })
}

fn desugar_field(
    member: Member,
    var_pat: Pat,
    temp_name: &Ident,
    context: &mut Context,
) -> syn::Result<Statement> {
    let temp_var = Expression::Variable(context.variable(temp_name).unwrap());
    let is_const = temp_var.is_const();
    let init = Some(Box::new(Expression::FieldAccess {
        base: Box::new(temp_var),
        field: member,
    }));
    let Pattern {
        ident,
        ty,
        is_ref,
        is_mut,
    } = parse_pat(var_pat.clone())?;
    let variable = context.push_variable(ident, ty, is_const, is_ref, is_mut);
    let statement = Statement::Local { variable, init };
    Ok(statement)
}

fn create_local(name: Ident, init: Expression, context: &mut Context) -> Statement {
    let variable = context.push_variable(name, init.ty(), init.is_const(), false, false);
    let init = Some(Box::new(init));

    Statement::Local { variable, init }
}
