use proc_macro2::{Span, TokenStream};
use quote::{quote, quote_spanned, ToTokens};
use syn::{spanned::Spanned, Pat, Token};

use crate::{
    expression::Expression,
    statement::{parse_pat, Statement},
};

impl ToTokens for Statement {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
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
                let mutable = mutable.then(|| quote![mut]);
                let as_const = init.as_ref().and_then(|init| init.as_const());
                if as_const.is_some() && mutable.is_some() {
                    let init = as_const.unwrap();
                    quote_spanned! {*span=>
                        let #name = #init;
                    }
                } else if let Some(init) = init {
                    quote_spanned! {*span=>
                        let #mutable #name = #init;
                    }
                } else {
                    quote_spanned! {*span=>
                        let #mutable #name: #ty;
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
                let terminator = terminated.then(|| Token![;](*span));
                if let Some(as_const) = expression.as_const() {
                    quote![#as_const #terminator]
                } else {
                    quote_spanned! {*span=>
                        #expression #terminator
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
