use quote::{quote, quote_spanned, ToTokens};
use syn::spanned::Spanned;

use crate::{
    expression::Expression, generate::expression::generate_var, ir_type, statement::Statement,
};

impl ToTokens for Statement {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let statement = ir_type(("Statement"));
        let expr = ir_type(("Expr"));

        let out = match self {
            Statement::Local {
                left,
                init,
                mutable,
                span,
                ty,
            } => {
                let span = span.clone();

                let name = match &**left {
                    Expression::Variable { name, .. } => name,
                    Expression::Init { left, .. } => match &**left {
                        Expression::Variable { name, .. } => name,
                        _ => panic!("Init left is always variable"),
                    },
                    _ => panic!("Local is always variable or init"),
                };
                let as_const = init.as_ref().and_then(|init| init.as_const());
                if as_const.is_some() && !mutable {
                    let init = as_const.unwrap();
                    quote_spanned! {span=>
                        let #name = #init;
                    }
                } else {
                    // Separate init and declaration in case initializer uses an identically named
                    // variable that would be overwritten by the declaration.
                    let initializer = init.as_ref().map(|init| quote![let __init = #init;]);
                    let left = if let Some(init) = init {
                        let span = span.clone();
                        let init_ty = ir_type("Initializer");
                        quote_spanned! {span=>
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
                        generate_var(name, ty, span, vectorization);
                    let variable_decl = quote_spanned! {span=>
                        let #name = #variable;
                    };

                    let ty = if let Some(ty) = ty {
                        let span = ty.span();
                        let sq_type = ir_type(("SquareType"));
                        quote_spanned! {span=>
                            Some(<#ty as #sq_type>::ir_type())
                        }
                    } else {
                        quote![None]
                    };

                    quote_spanned! {span=>
                        #initializer
                        #variable_decl
                        __statements.push({
                                #statement::Local {
                                variable: Box::new(#expr::expression_untyped(&#left)),
                                mutable: #mutable,
                                ty: #ty
                            }
                        });
                    }
                }
            }
            Statement::Expression {
                expression,
                terminated,
                span,
            } => {
                let span = span.clone();
                if *terminated {
                    quote_spanned! {span=>
                        __statements.push(#statement::Expression(
                            Box::new(#expr::expression_untyped(&#expression))
                        ));
                    }
                } else {
                    quote_spanned! {span=>
                        __statements.push(#statement::Return(
                            Box::new(#expr::expression_untyped(&#expression))
                        ));
                    }
                }
            }
        };

        tokens.extend(out);
    }
}
