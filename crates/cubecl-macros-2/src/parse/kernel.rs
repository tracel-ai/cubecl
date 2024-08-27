use syn::{parse_quote, Attribute, FnArg, Generics, Ident, ItemFn, Pat, Type, Visibility};

use crate::{expression::Expression, scope::Context};

use super::{branch::parse_block, helpers::is_comptime_attr};

pub struct Kernel {
    pub(crate) visibility: Visibility,
    pub(crate) name: Ident,
    pub(crate) parameters: Vec<(Ident, Type, bool)>,
    pub(crate) block: Expression,
    pub(crate) returns: Type,
    pub(crate) generics: Generics,
}

impl Kernel {
    pub fn from_item_fn(function: ItemFn) -> syn::Result<Self> {
        let name = function.sig.ident;
        let vis = function.vis;
        let generics = function.sig.generics;
        let returns = match function.sig.output {
            syn::ReturnType::Default => parse_quote![()],
            syn::ReturnType::Type(_, ty) => *ty,
        };
        let mut context = Context::new(returns.clone());
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
        let block = parse_block(*function.block, &mut context)?;
        context.pop_scope(); // Pop function local scope

        Ok(Kernel {
            visibility: vis,
            generics,
            name,
            parameters: variables,
            block,
            returns,
        })
    }
}

fn is_const(attrs: &[Attribute]) -> bool {
    attrs.iter().any(is_comptime_attr)
}
