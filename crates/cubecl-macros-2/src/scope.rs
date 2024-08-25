use std::{collections::HashMap, num::NonZero};

use proc_macro2::TokenStream;
use quote::{format_ident, quote, quote_spanned};
use syn::{spanned::Spanned, Ident, Type};

use crate::generate::expression::generate_var;

pub const KEYWORDS: [&str; 21] = [
    "ABSOLUTE_POS",
    "ABSOLUTE_POS_X",
    "ABSOLUTE_POS_Y",
    "ABSOLUTE_POS_Z",
    "UNIT_POS",
    "UNIT_POS_X",
    "UNIT_POS_Y",
    "UNIT_POS_Z",
    "CUBE_POS",
    "CUBE_POS_X",
    "CUBE_POS_Y",
    "CUBE_POS_Z",
    "CUBE_DIM",
    "CUBE_DIM_X",
    "CUBE_DIM_Y",
    "CUBE_DIM_Z",
    "CUBE_COUNT",
    "CUBE_COUNT_X",
    "CUBE_COUNT_Y",
    "CUBE_COUNT_Z",
    "SUBCUBE_DIM",
];

pub struct Context {
    pub return_type: Type,
    scopes: Vec<Scope>,
    // Allows for global variable analysis
    scope_history: Vec<Scope>,
}

impl Context {
    pub fn new(return_type: Type) -> Self {
        let mut root_scope = Scope::default();
        root_scope.variables.extend(KEYWORDS.iter().map(|it| {
            let name = format_ident!("{it}");
            let tokens = quote![u32];
            let ty = syn::parse2(tokens).unwrap();
            ManagedVar {
                name,
                ty: Some(ty),
                is_const: false,
            }
        }));
        Self {
            return_type,
            scopes: vec![root_scope],
            scope_history: Default::default(),
        }
    }

    pub fn push_variable(&mut self, name: Ident, ty: Option<Type>, is_const: bool) {
        self.scopes
            .last_mut()
            .expect("Scopes must at least have root scope")
            .variables
            .push(ManagedVar { name, ty, is_const });
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(Scope::default())
    }

    pub fn pop_scope(&mut self) {
        let scope = self.scopes.pop().expect("Can't pop root scope");
        self.scope_history.push(scope);
    }

    pub fn with_scope<T>(&mut self, with: impl FnOnce(&mut Self) -> T) -> T {
        self.push_scope();
        let res = with(self);
        self.pop_scope();
        res
    }

    pub fn restore_scope(&mut self) {
        let scope = self.scope_history.pop();
        if let Some(scope) = scope {
            self.scopes.push(scope);
        }
    }

    pub fn current_scope(&self) -> &Scope {
        self.scopes
            .last()
            .expect("Scopes must at least have root scope")
    }

    pub fn variable(&self, name: &Ident) -> Option<ManagedVar> {
        // Walk through each scope backwards until we find the variable.
        self.scopes
            .iter()
            .rev()
            .flat_map(|scope| scope.variables.iter().rev())
            .find(|var| name == &var.name)
            .cloned()
    }

    pub fn extend(&mut self, vars: impl IntoIterator<Item = (Ident, Option<Type>, bool)>) {
        self.scopes
            .last_mut()
            .expect("Scopes must at least have root scope")
            .variables
            .extend(
                vars.into_iter()
                    .map(|(name, ty, is_const)| ManagedVar { name, ty, is_const }),
            )
    }
}

#[derive(Default)]
pub struct Scope {
    variables: Vec<ManagedVar>,
}

#[derive(Clone)]
pub struct ManagedVar {
    pub name: Ident,
    pub ty: Option<Type>,
    pub is_const: bool,
}

impl Scope {
    pub fn generate_vars(&self) -> Vec<TokenStream> {
        self.variables
            .iter()
            .map(|ManagedVar { name, ty, .. }| {
                let mut span = name.span();
                let var = generate_var(name, ty, span, None);
                quote_spanned! {span=>
                    let #name = #var;
                }
            })
            .collect()
    }
}
