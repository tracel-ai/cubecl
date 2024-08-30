use proc_macro2::{Span, TokenStream};
use quote::{quote, quote_spanned, ToTokens};
use syn::{spanned::Spanned, Pat, Token};

use crate::{
    expression::Expression,
    generate::expression::generate_var,
    ir_type,
    statement::{parse_pat, Statement},
};

impl ToTokens for Statement {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let statement = ir_type("Statement");
        let expr = ir_type("Expr");

        let out = match self {
            Statement::Local {
                left,
                init,
                mutable,
                span,
                ty,
            } => {
                let name = match &**left {
                    Expression::Variable { name, .. } => name,
                    _ => panic!("Local is always variable or init"),
                };
                let as_const = init.as_ref().and_then(|init| init.as_const());
                if as_const.is_some() && !mutable {
                    let init = as_const.unwrap();
                    quote_spanned! {*span=>
                        let #name = #init;
                    }
                } else {
                    // Separate init and declaration in case initializer uses an identically named
                    // variable that would be overwritten by the declaration.
                    let initializer = init.as_ref().map(|init| quote![let __init = #init;]);
                    let left = if init.is_some() {
                        let init_ty = ir_type("Initializer");
                        quote_spanned! {*span=>
                            #init_ty {
                                left: #name,
                                right: __init
                            }
                        }
                    } else {
                        quote![#name]
                    };
                    let expr = ir_type("Expr");
                    let vectorization = initializer
                        .is_some()
                        .then(|| quote![#expr::vectorization(&__init)]);
                    let variable: proc_macro2::TokenStream =
                        generate_var(name, ty, *span, vectorization);
                    let variable_decl = quote_spanned! {*span=>
                        let #name = #variable;
                    };

                    let ty = if let Some(ty) = ty {
                        let span = ty.span();
                        let sq_type = ir_type("SquareType");
                        quote_spanned! {span=>
                            Some(<#ty as #sq_type>::ir_type())
                        }
                    } else {
                        quote![None]
                    };

                    quote_spanned! {*span=>
                        #initializer
                        #variable_decl
                        __statements.push({
                            #statement::Local {
                                variable: #expr::expression_untyped(&(#left)),
                                mutable: #mutable,
                                ty: #ty
                            }
                        });
                    }
                }
            }
            Statement::Destructure { fields, span } => {
                let fields = generate_struct_destructure(fields, *span);
                match fields {
                    Ok(fields) => fields,
                    Err(e) => e.to_compile_error(),
                }
            }
            Statement::Expression {
                expression,
                span,
                terminated,
            } => {
                if let Some(as_const) = expression.as_const() {
                    let terminator = terminated.then(|| Token![;](*span));
                    quote![#as_const #terminator]
                } else {
                    quote_spanned! {*span=>
                        __statements.push(#statement::Expression(
                            #expr::expression_untyped(&(#expression))
                        ));
                    }
                }
            }
            Statement::Skip => TokenStream::new(),
        };

        tokens.extend(out);
    }
}

fn generate_struct_destructure(
    fields: &[(Pat, Expression)],
    span: Span,
) -> syn::Result<TokenStream> {
    let fields = fields
        .iter()
        .map(|(pat, init)| {
            let span = pat.span();
            let (ident, ty, mutable) = parse_pat(pat.clone())?;
            let statement = Statement::Local {
                left: Box::new(Expression::Variable {
                    name: ident,
                    ty: None,
                    span,
                }),
                init: Some(Box::new(init.clone())),
                mutable,
                ty,
                span,
            };
            Ok(quote![#statement])
        })
        .collect::<syn::Result<Vec<_>>>()?;

    Ok(quote_spanned! {span=>
        #(#fields)*
    })
}
