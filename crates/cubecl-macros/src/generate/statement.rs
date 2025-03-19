use proc_macro2::{Span, TokenStream};
use quote::{quote, quote_spanned};
use syn::{Token, spanned::Spanned};

use crate::{expression::Expression, paths::frontend_type, scope::Context, statement::Statement};

impl Statement {
    pub fn to_tokens(&self, context: &mut Context) -> TokenStream {
        match self {
            Statement::Local { variable, init } => {
                let cube_type = frontend_type("CubeType");
                let debug_var = frontend_type("debug_var_expand");
                let name = &variable.name;
                let is_mut = variable.is_mut || init.as_deref().map(is_mut_owned).unwrap_or(false);
                let mutable = variable.is_mut.then(|| quote![mut]);
                let is_const = init
                    .as_ref()
                    .map(|it| it.as_const(context).is_some())
                    .unwrap_or(false);
                let init = if is_mut {
                    if let Some(as_const) =
                        init.as_ref().and_then(|it| it.as_const_primitive(context))
                    {
                        let expand = frontend_type("ExpandElementTyped");
                        Some(
                            quote_spanned![as_const.span()=> #expand::from_lit(context, #as_const)],
                        )
                    } else if let Some(as_const) = init.as_ref().and_then(|it| it.as_const(context))
                    {
                        Some(quote_spanned![as_const.span()=> #as_const.clone()])
                    } else {
                        init.as_ref().map(|it| it.to_tokens(context))
                    }
                } else {
                    init.as_ref().map(|init| {
                        init.as_const(context)
                            .unwrap_or_else(|| init.to_tokens(context))
                    })
                };
                let ty = variable.ty.as_ref().map(|ty| {
                    quote_spanned! {
                        ty.span()=> :<#ty as #cube_type>::ExpandType
                    }
                });

                let init = match (is_mut, init) {
                    (true, Some(init)) => {
                        let init_ty = frontend_type("Init");
                        let init_ty = quote_spanned![init.span()=> #init_ty::init(_init, context)];
                        Some(quote! {
                            {
                                let _init = #init;
                                #init_ty
                            }
                        })
                    }
                    (_, init) => init,
                };

                if let Some(mut init) = init {
                    if is_mut || !is_const {
                        let name_str = name.to_string();
                        init = quote! {{
                            let __init = #init;
                            #debug_var(context, #name_str, __init)
                        }};
                    }

                    quote![let #mutable #name #ty = #init;]
                } else {
                    quote![let #mutable #name #ty;]
                }
            }
            Statement::Expression {
                expression,
                terminated,
            } => {
                let terminator = terminated.then(|| Token![;](Span::call_site()));
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
        Expression::Variable(var) => var.is_mut && !var.is_ref,
        Expression::FieldAccess { base, .. } => is_mut_owned(base),
        _ => false,
    }
}
