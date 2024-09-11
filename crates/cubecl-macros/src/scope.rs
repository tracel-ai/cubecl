use std::{
    mem::replace,
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
    mut_scope_idx: usize,
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
            mut_scope_idx: 0,
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

    fn pop_scope(&mut self) -> Scope {
        self.scopes.pop().expect("Can't pop root scope")
    }

    pub fn with_scope<T>(&mut self, with: impl FnOnce(&mut Self) -> T) -> (T, Scope) {
        self.push_scope();
        let res = with(self);
        let scope = self.pop_scope();
        (res, scope)
    }

    pub fn with_restored_scope<T>(
        &mut self,
        scope: &Scope,
        with: impl FnOnce(&mut Self) -> T,
    ) -> T {
        self.scopes.push(scope.deep_clone());
        let res = with(self);
        self.pop_scope();
        res
    }

    /// Mutable closures (for loops) have different behaviour because outer vars must be cloned
    pub fn with_restored_closure_scope<T>(
        &mut self,
        scope: &Scope,
        with: impl FnOnce(&mut Self) -> T,
    ) -> T {
        let old_mut_idx = replace(&mut self.mut_scope_idx, self.scopes.len());
        self.scopes.push(scope.deep_clone());
        let res = with(self);
        self.pop_scope();
        self.mut_scope_idx = old_mut_idx;
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
        let (level, var) = self
            .scopes
            .iter()
            .enumerate()
            .flat_map(|(i, scope)| scope.variables.iter().map(move |it| (i, it)))
            .find(|(_, var)| &var.name == name && var.use_count.load(Ordering::Acquire) > 0)
            .unwrap_or_else(|| {
                panic!(
                    "Trying to get use count of variable {name} that never existed.\nScopes: {:#?}\n",
                    self.scopes,
                );
            });
        let count = var.use_count.fetch_sub(1, Ordering::AcqRel);
        level >= self.mut_scope_idx && count <= 1
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
}

impl Scope {
    pub fn deep_clone(&self) -> Self {
        Scope {
            variables: self
                .variables
                .iter()
                .map(|var| ManagedVar {
                    use_count: AtomicUsize::new(var.use_count.load(Ordering::Acquire)).into(),
                    ..var.clone()
                })
                .collect(),
        }
    }
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
