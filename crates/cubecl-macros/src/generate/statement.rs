use proc_macro2::TokenStream;
use quote::{quote, quote_spanned};
use syn::{spanned::Spanned, Pat, Token};

use crate::{
    expression::Expression,
    paths::frontend_type,
    scope::Context,
    statement::{parse_pat, Statement},
};

impl Statement {
    pub fn to_tokens(&self, context: &mut Context) -> TokenStream {
        match self {
            Statement::Local {
                left,
                init,
                mutable,
                ty,
            } => {
                let name = match &**left {
                    Expression::Variable { name, .. } => name,
                    _ => panic!("Local is always variable or init"),
                };
                let mutable = mutable.then(|| quote![mut]);
                let init_span = init.as_ref().map(|it| it.span());
                let init = if mutable.is_some() {
                    if let Some(as_const) = init.as_ref().and_then(|it| it.as_const(context)) {
                        let expand = frontend_type("ExpandElementTyped");
                        Some(quote_spanned![as_const.span()=> #expand::from_lit(#as_const)])
                    } else {
                        init.as_ref().map(|it| it.to_tokens(context))
                    }
                } else {
                    init.as_ref().map(|init| {
                        init.as_const(context)
                            .unwrap_or_else(|| init.to_tokens(context))
                    })
                };

                let init = match (mutable.is_some(), init) {
                    (true, Some(init)) => {
                        let init_ty = frontend_type("Init");
                        let init_ty =
                            quote_spanned![init_span.unwrap()=> #init_ty::init(_init, context)];
                        Some(quote! {
                            {
                                let _init = #init;
                                #init_ty
                            }
                        })
                    }
                    (_, init) => init,
                };

                if let Some(init) = init {
                    quote![let #mutable #name = #init;]
                } else {
                    quote![let #mutable #name: #ty;]
                }
            }
            Statement::Destructure { fields } => {
                let fields = generate_struct_destructure(fields, context);
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
                if let Some(as_const) = expression.as_const(context) {
                    quote![#as_const #terminator]
                } else {
                    let expression = expression.to_tokens(context);
                    quote![#expression #terminator]
                }
            }
            Statement::Skip => TokenStream::new(),
        }
    }
}

fn generate_struct_destructure(
    fields: &[(Pat, Expression)],
    context: &mut Context,
) -> syn::Result<TokenStream> {
    let fields = fields
        .iter()
        .map(|(pat, init)| {
            let (ident, ty, mutable) = parse_pat(pat.clone())?;
            let statement = Statement::Local {
                left: Box::new(Expression::Variable {
                    name: ident,
                    ty: None,
                }),
                init: Some(Box::new(init.clone())),
                mutable,
                ty,
            };
            let statement = statement.to_tokens(context);
            Ok(quote![#statement])
        })
        .collect::<syn::Result<Vec<_>>>()?;

    Ok(quote! {span=>
        #(#fields)*
    })
}
