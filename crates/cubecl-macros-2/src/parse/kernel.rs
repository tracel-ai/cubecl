use std::cell::RefCell;

use quote::{format_ident, quote};
use syn::{parse::Parse, Attribute, FnArg, Generics, Ident, ItemFn, Meta, Pat, Type, Visibility};

use crate::{scope::Context, statement::Statement};

use super::helpers::is_comptime_attr;

pub struct Kernel {
    pub(crate) visibility: Visibility,
    pub(crate) name: Ident,
    pub(crate) parameters: Vec<(Ident, Type, bool)>,
    pub(crate) statements: Vec<Statement>,
    pub(crate) returns: Type,
    pub(crate) generics: Generics,

    pub(crate) context: RefCell<Context>,
}

impl Kernel {
    pub fn from_item_fn(function: ItemFn) -> syn::Result<Self> {
        let mut context = Context::default();

        let name = function.sig.ident;
        let vis = function.vis;
        let generics = function.sig.generics;
        let returns = match function.sig.output {
            syn::ReturnType::Default => syn::parse2(quote![()]).unwrap(),
            syn::ReturnType::Type(_, ty) => *ty,
        };
        let parameters = function
            .sig
            .inputs
            .into_iter()
            .map(|input| match &input {
                FnArg::Typed(arg) => Ok(arg.clone()),
                _ => Err(syn::Error::new_spanned(
                    input,
                    "Unsupported input for kernel",
                )),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let variables = parameters
            .into_iter()
            .map(|input| -> syn::Result<(Ident, Type, bool)> {
                let ty = *input.ty;
                let ident = match *input.pat {
                    Pat::Ident(ident) => ident.ident,
                    input => Err(syn::Error::new_spanned(
                        input,
                        "kernel input should be ident",
                    ))?,
                };
                let is_const = is_const(&input.attrs);
                Ok((ident, ty, is_const))
            })
            .collect::<Result<Vec<_>, _>>()?;

        context.extend(
            variables
                .iter()
                .cloned()
                .map(|(ident, ty, is_const)| (ident, Some(ty), is_const)),
        );
        context.push_scope(); // Push function local scope

        let statements = function
            .block
            .stmts
            .into_iter()
            .map(|statement| Statement::from_stmt(statement, &mut context))
            .collect::<Result<Vec<_>, _>>()?;

        context.pop_scope(); // Pop function local scope

        Ok(Kernel {
            visibility: vis,
            generics,
            name,
            parameters: variables,
            statements,
            context: RefCell::new(context),
            returns,
        })
    }
}

fn is_const(attrs: &[Attribute]) -> bool {
    attrs.iter().any(is_comptime_attr)
}
