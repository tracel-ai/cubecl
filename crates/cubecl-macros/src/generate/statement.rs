use proc_macro2::TokenStream;
use quote::{quote, quote_spanned};
use syn::{spanned::Spanned, Token};

use crate::{expression::Expression, paths::frontend_type, scope::Context, statement::Statement};

impl Statement {
    pub fn to_tokens(&self, context: &mut Context) -> TokenStream {
        match self {
            Statement::Local {
                left,
                init,
                mutable,
                ty,
            } => {
                let cube_type = frontend_type("CubeType");
                let name = match &**left {
                    Expression::Variable { name, .. } => name,
                    _ => panic!("Local is always variable or init"),
                };
                let is_mut = *mutable || init.as_deref().map(is_mut_owned).unwrap_or(false);
                let mutable = mutable.then(|| quote![mut]);
                let init_span = init.as_ref().map(|it| it.span());
                let init = if is_mut {
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
                let ty = ty
                    .as_ref()
                    .map(|ty| quote_spanned![ty.span()=> :<#ty as #cube_type>::ExpandType]);

                let init = match (is_mut, init) {
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
                    quote![let #mutable #name #ty = #init;]
                } else {
                    quote![let #mutable #name #ty;]
                }
            }
            Statement::Group { statements } => {
                let statements = statements.iter().map(|it| it.to_tokens(context));
                quote! {
                    #(#statements)*
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

fn is_mut_owned(init: &Expression) -> bool {
    match init {
        Expression::Variable { is_ref, is_mut, .. } => *is_mut && !is_ref,
        Expression::FieldAccess { base, .. } => is_mut_owned(base),
        _ => false,
    }
}
