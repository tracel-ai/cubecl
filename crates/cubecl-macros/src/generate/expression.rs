use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote, quote_spanned, ToTokens};
use syn::{spanned::Spanned, Ident, PathArguments, Type};

use crate::{
    expression::{Block, Expression},
    ir_type, prefix_ir,
};

impl ToTokens for Expression {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let out = match self {
            Expression::Binary {
                left,
                operator,
                right,
                span,
                ..
            } => {
                let expr_ty = prefix_ir(format_ident!("{}Expr", operator.to_string()));
                quote_spanned! {*span=>
                    #expr_ty::new(
                        #left,
                        #right
                    )
                }
            }
            Expression::Unary {
                input,
                operator,
                span,
                ..
            } => {
                let ty = prefix_ir(format_ident!("{}Expr", operator.to_string()));
                quote_spanned! {*span=>
                    #ty::new(
                        #input,
                    )
                }
            }
            Expression::Variable { name, span, .. } => {
                quote_spanned! {*span=>
                    #name.clone()
                }
            }
            Expression::FieldAccess {
                base, field, span, ..
            } => {
                let field = match field {
                    syn::Member::Named(ident) => format_ident!("__{ident}"),
                    syn::Member::Unnamed(index) => format_ident!("__{}", index.index),
                };
                quote_spanned! {*span=>
                    #base.expand().#field()
                }
            }
            Expression::Literal { value, span, .. } => {
                quote_spanned! {*span=>
                    #value
                }
            }
            Expression::Assigment {
                left, right, span, ..
            } => {
                let ty = prefix_ir(format_ident!("Assignment"));
                quote_spanned! {*span=>
                    #ty {
                        left: #left,
                        right: #right
                    }
                }
            }
            Expression::Verbatim { tokens, .. } => {
                let span = tokens.span();
                quote_spanned! {span=>
                    #tokens
                }
            }
            Expression::Block(block) => block.to_token_stream(),
            Expression::FunctionCall {
                func,
                span,
                args,
                associated_type,
            } => {
                let args: Vec<TokenStream> = if self.is_const() {
                    args.iter().map(|arg| arg.to_token_stream()).collect()
                } else {
                    let once_expr = ir_type("OnceExpr");
                    args.iter()
                        .map(|arg| {
                            if arg.is_const() {
                                arg.to_token_stream()
                            } else {
                                quote![#once_expr::new(#arg)]
                            }
                        })
                        .collect()
                };

                // We pass in the `Variable`s and `Literal`s into the expansion so they can be rebound
                // in the function root scope
                if let Some((ty_path, name)) = associated_type {
                    let static_expand = ir_type("StaticExpand");
                    quote_spanned! {*span=>
                        <#ty_path as #static_expand>::Expanded::#name(#(#args),*)
                    }
                } else {
                    let (generics, path) = split_generics(func);
                    quote_spanned! {*span=>
                        #path::expand #generics(#(#args),*)
                    }
                }
            }
            Expression::MethodCall {
                receiver,
                method,
                args,
                span,
            } => {
                let expand = if receiver.is_const() {
                    format_ident!("partial_expand")
                } else {
                    format_ident!("expand")
                };
                quote_spanned! {*span=>
                    #receiver.#expand().#method(#(#args),*)
                }
            }
            Expression::Break { span } => {
                let brk = ir_type("Break");
                quote_spanned! {*span=>
                    #brk
                }
            }
            Expression::Cast { from, to, span } => {
                let cast = ir_type("Cast");
                quote_spanned! {*span=>
                    #cast::<_, #to>::new(#from)
                }
            }
            Expression::Continue { span } => {
                let cont = ir_type("Continue");
                quote_spanned! {*span=>
                    #cont
                }
            }
            Expression::ForLoop {
                range,
                unroll,
                var_name,
                var_ty,
                block,
                span,
            } => {
                let variable = generate_var(var_name, true, var_ty, *span, None);
                let for_ty = ir_type("ForLoop");

                if let Some(unroll) = unroll {
                    //let unrolled = generate_unroll(block, range, var_name);
                    quote_spanned! {*span=>
                        {
                            let #var_name = #variable;
                            if #unroll {
                                #for_ty::new_unroll(#range, #var_name.clone(), #block)
                            } else {
                                #for_ty::new(#range, #var_name.clone(), #block)
                            }
                        }
                    }
                } else {
                    quote_spanned! {*span=>
                        {
                            let #var_name = #variable;
                            #for_ty::new(#range, #var_name.clone(), #block)
                        }
                    }
                }
            }
            Expression::WhileLoop {
                condition,
                block,
                span,
            } => {
                let while_ty = ir_type("WhileLoop");

                quote_spanned! {*span=>
                    {
                        #while_ty::new(#condition, #block)
                    }
                }
            }
            Expression::Loop { block, span } => {
                let loop_ty = ir_type("Loop");

                quote_spanned! {*span=>
                    {
                        #loop_ty::new(#block)
                    }
                }
            }
            Expression::If {
                condition,
                then_block,
                else_branch,
                span,
            } => {
                let if_ty = ir_type("If");

                if let Some(as_const) = condition.as_const() {
                    let else_branch = else_branch.as_ref().map(|it| {
                        quote! {
                            else {
                                #it
                            }
                        }
                    });
                    quote_spanned! {*span=>
                        if #as_const {
                            #then_block
                        } #else_branch
                    }
                } else {
                    let else_branch = else_branch
                        .as_ref()
                        .map(|it| quote![Some(#it)])
                        .unwrap_or_else(|| quote![None::<()>]);
                    quote_spanned! {*span=>
                        #if_ty::new(#condition, #then_block, #else_branch)
                    }
                }
            }
            Expression::ConstVariable { name, .. } => quote![#name],
            Expression::Path { path, .. } => quote![#path],
            Expression::Range {
                start,
                end,
                inclusive,
                span,
            } => {
                if let Some(end) = end {
                    let range = ir_type("RangeExpr");
                    quote_spanned! {*span=>
                        #range::new(#start, #end, #inclusive)
                    }
                } else {
                    let range = ir_type("SliceRangeExpr");
                    let end = end
                        .as_ref()
                        .map(|it| quote![Some(Box::new(#it))])
                        .unwrap_or_else(|| quote![None]);
                    quote_spanned! {*span=>
                        #range::new(#start, #end, #inclusive)
                    }
                }
            }
            Expression::Return { expr, ty, span } => {
                let ret_ty = ir_type("Return");
                let ty = expr
                    .as_ref()
                    .map(|_| quote![::<#ty, _>])
                    .unwrap_or_else(|| quote![::<(), ()>]);
                let ret_expr = expr
                    .as_ref()
                    .map(|it| quote![Some(#it)])
                    .unwrap_or_else(|| quote![None]);
                quote_spanned! {*span=>
                    #ret_ty #ty::new(#ret_expr)
                }
            }
            Expression::Array { span, .. } => {
                if let Some(constant) = self.as_const() {
                    constant
                } else {
                    syn::Error::new(*span, "Array expressions can't be used at runtime")
                        .to_compile_error()
                }
            }
            Expression::Tuple { span, .. } => {
                if let Some(constant) = self.as_const() {
                    constant
                } else {
                    syn::Error::new(*span, "Tuple expressions can't be used at runtime")
                        .to_compile_error()
                }
            }
            Expression::Index { expr, index, span } => {
                quote_spanned! {*span=>
                    #expr.expand().index(#index)
                }
            }
            Expression::Slice { expr, ranges, span } => {
                let range_ty = ir_type("SliceRangeExpr");
                quote_spanned! {*span=>
                    #expr.expand().slice(vec![#(Box::new(#range_ty::from(#ranges))),*])
                }
            }
            Expression::ArrayInit { init, len, span } => {
                let init_ty = ir_type("ArrayInit");
                quote_spanned! {*span=>
                    #init_ty::new(#len, #init)
                }
            }
            Expression::VerbatimTerminated { tokens } => tokens.clone(),
            Expression::Reference { inner } => {
                if let Some(as_const) = inner.as_const() {
                    quote![&#as_const]
                } else {
                    quote![#inner]
                }
            }
            Expression::StructInit { path, fields } => {
                let cube_type = ir_type("CubeType");

                quote! {
                    <#path as #cube_type>::Runtime::new(#(#fields),*)
                }
            }
            Expression::Closure { tokens } => tokens.clone(),
        };

        tokens.extend(out);
    }
}

impl ToTokens for Block {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let block = ir_type("BlockExpr");
        let ret = self
            .ret
            .as_ref()
            .map(|ret| quote![#ret])
            .unwrap_or_else(|| quote![()]);
        let inner = &self.inner;
        tokens.extend(quote_spanned! {self.span=>
            {
                let mut __statements = Vec::new();
                #(#inner)*
                #block::new(__statements, #ret)
            }
        });
    }
}

pub fn generate_var(
    name: &Ident,
    mutable: bool,
    ty: &Option<Type>,
    span: Span,
    vectorization: Option<TokenStream>,
) -> TokenStream {
    let var = ir_type("Variable");
    let name = name.to_token_stream().to_string();
    let ty = ty.as_ref().map(|ty| {
        quote_spanned! {ty.span()=>
            ::<#ty>
        }
    });
    let vectorization = vectorization.unwrap_or(quote![None]);
    quote_spanned! {span=>
        #var #ty ::new(#name, #mutable, #vectorization)
    }
}

fn split_generics(path: &Expression) -> (PathArguments, TokenStream) {
    let mut path = match path {
        Expression::Path { path, .. } => path.clone(),
        _ => return (PathArguments::None, quote![#path]),
    };
    let generics = if let Some(last) = path.segments.last_mut() {
        core::mem::replace(&mut last.arguments, PathArguments::None)
    } else {
        PathArguments::None
    };
    (generics, quote![#path])
}

// fn generate_unroll(block: &Block, range: &Expression, var: &Ident) -> TokenStream {
//     let ret = block.ret.as_ref().map(|ret| Statement::Expression {
//         expression: ret.clone(),
//         terminated: true,
//         span: ret.span(),
//     });

//     let inner = &block.inner;

//     let func = quote! {
//         #(#inner)*
//         #ret
//     };

//     let block = ir_type("BlockExpr");
//     let for_range = ir_type("ForLoopRange");
//     quote! {
//         let (__start, __end, __step, __inclusive) = #for_range::as_primitive(&(#range));
//         let mut __statements = Vec::new();

//         match (__step, __inclusive) {
//             (None, true) => {
//                 for #var in __start..=__end {
//                     #func
//                 }
//             }
//             (None, false) => {
//                 for #var in __start..__end {
//                     #func
//                 }
//             }
//             (Some(step), true) => {
//                 for #var in (__start..=__end).step_by(__step) {
//                     #func
//                 }
//             }
//             (Some(step), false) => {
//                 for #var in (__start..__end).step_by(__step) {
//                     #func
//                 }
//             }
//         };

//         #block::new(__statements, ())
//     }
// }
