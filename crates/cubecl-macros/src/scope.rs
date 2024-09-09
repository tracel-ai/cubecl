use std::{
    collections::{HashMap, VecDeque},
    rc::Rc,
    sync::atomic::{AtomicUsize, Ordering},
};

use quote::format_ident;
use syn::{parse_quote, Ident, Type};

use crate::parse::kernel::KernelParam;

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

#[derive(Clone)]
pub struct Context {
    pub return_type: Type,
    scopes: Vec<Scope>,
    // Allows for global variable analysis
    scope_history: HashMap<usize, VecDeque<Scope>>,
}

impl Context {
    pub fn new(return_type: Type) -> Self {
        let mut root_scope = Scope::default();
        root_scope.variables.extend(KEYWORDS.iter().map(|it| {
            let name = format_ident!("{it}");
            let ty = parse_quote![u32];
            ManagedVar {
                name,
                ty: Some(ty),
                is_const: false,
                is_ref: false,
                is_mut: false,
                is_keyword: true,
                use_count: AtomicUsize::new(0).into(),
            }
        }));
        Self {
            return_type,
            scopes: vec![root_scope],
            scope_history: Default::default(),
        }
    }

    pub fn push_variable(
        &mut self,
        name: Ident,
        ty: Option<Type>,
        is_const: bool,
        is_ref: bool,
        is_mut: bool,
    ) {
        self.scopes
            .last_mut()
            .expect("Scopes must at least have root scope")
            .variables
            .push(ManagedVar {
                name,
                ty,
                is_const,
                is_ref,
                is_mut,
                is_keyword: false,
                use_count: AtomicUsize::new(0).into(),
            });
    }

    fn push_scope(&mut self) {
        self.scopes.push(Scope::default())
    }

    fn pop_scope(&mut self) {
        let scope = self.scopes.pop().expect("Can't pop root scope");
        self.scope_history
            .entry(self.scopes.len())
            .or_default()
            .push_back(scope);
    }

    fn delete_scope(&mut self) {
        self.scopes.pop();
    }

    pub fn with_scope<T>(&mut self, with: impl FnOnce(&mut Self) -> T) -> T {
        self.push_scope();
        let res = with(self);
        self.pop_scope();
        res
    }

    fn restore_scope(&mut self) {
        let scope = self
            .scope_history
            .get_mut(&(self.scopes.len()))
            .and_then(|it| it.pop_front());
        if let Some(scope) = scope {
            self.scopes.push(scope);
        }
    }

    fn restore_mut_scope(&mut self) {
        let scope = self
            .scope_history
            .get_mut(&(self.scopes.len()))
            .and_then(|it| it.pop_front());
        if let Some(mut scope) = scope {
            scope.is_mut = true;
            self.scopes.push(scope);
        }
    }

    pub fn with_restored_scope<T>(&mut self, with: impl FnOnce(&mut Self) -> T) -> T {
        self.restore_scope();
        let res = with(self);
        self.delete_scope();
        res
    }

    /// Mutable closures (for loops) have different behaviour because outer vars must be cloned
    pub fn with_restored_closure_scope<T>(&mut self, with: impl FnOnce(&mut Self) -> T) -> T {
        self.restore_mut_scope();
        let res = with(self);
        self.delete_scope();
        res
    }

    pub fn variable(&self, name: &Ident) -> Option<ManagedVar> {
        // Walk through each scope backwards until we find the variable.
        self.scopes
            .iter()
            .rev()
            .flat_map(|scope| scope.variables.iter().rev())
            .find(|var| name == &var.name)
            .map(|var| {
                var.use_count.fetch_add(1, Ordering::AcqRel);
                var.clone()
            })
    }

    pub fn try_consume(&self, name: &Ident) -> bool {
        // Find innermost closure scope if it exists
        let mut_scope_idx = self
            .scopes
            .iter()
            .enumerate()
            .rev()
            .find(|(_, scope)| scope.is_mut)
            .map(|(i, _)| i);
        let (level, var) = self
            .scopes
            .iter()
            .enumerate()
            .rev()
            .flat_map(|(i, scope)| scope.variables.iter().map(move |it| (i, it)))
            .find(|(_, var)| &var.name == name && var.use_count.load(Ordering::Acquire) > 0)
            .unwrap_or_else(|| {
                panic!(
                    "Trying to get use count of variable {name} that never existed.\nScopes: {:#?}\nHistory:{:#?}",
                    self.scopes,
                    self.scope_history
                );
            });
        let count = var.use_count.fetch_sub(1, Ordering::AcqRel);
        /* if level == 0 {
            // Always clone outer vars since we can't see whether they're still used outside the
            // function
            false
        } else */
        if let Some(mut_scope_idx) = mut_scope_idx {
            // Always clone vars from outside closure, otherwise proceed as normal
            level >= mut_scope_idx && count <= 1
        } else {
            count <= 1
        }
    }

    pub fn extend(&mut self, vars: impl IntoIterator<Item = KernelParam>) {
        self.scopes
            .last_mut()
            .expect("Scopes must at least have root scope")
            .variables
            .extend(vars.into_iter().map(Into::into))
    }
}

#[derive(Default, Clone, Debug)]
pub struct Scope {
    variables: Vec<ManagedVar>,
    /// Must clone outer vars
    is_mut: bool,
}

#[derive(Clone, Debug)]
pub struct ManagedVar {
    pub name: Ident,
    pub ty: Option<Type>,
    pub is_const: bool,
    pub is_ref: bool,
    pub is_mut: bool,
    pub is_keyword: bool,
    pub use_count: Rc<AtomicUsize>,
}

impl From<KernelParam> for ManagedVar {
    fn from(value: KernelParam) -> Self {
        ManagedVar {
            name: value.name,
            ty: Some(value.ty),
            is_const: value.is_const,
            is_keyword: false,
            use_count: AtomicUsize::new(0).into(),
            is_ref: value.is_ref,
            is_mut: value.is_mut,
        }
    }
}
