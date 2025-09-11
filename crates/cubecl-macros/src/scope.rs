use std::{
    mem::replace,
    rc::Rc,
    sync::atomic::{AtomicUsize, Ordering},
};

use quote::format_ident;
use syn::{Ident, Type, parse_quote};

use crate::parse::kernel::KernelParam;

pub const KEYWORDS: [&str; 30] = [
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
    "PLANE_DIM",
    "UNIT_POS_PLANE",
    "CUBE_CLUSTER_DIM",
    "CUBE_CLUSTER_DIM_X",
    "CUBE_CLUSTER_DIM_Y",
    "CUBE_CLUSTER_DIM_Z",
    "CUBE_POS_CLUSTER",
    "CUBE_POS_CLUSTER_X",
    "CUBE_POS_CLUSTER_Y",
    "CUBE_POS_CLUSTER_Z",
];

pub type Scope = usize;
type ManagedScope = Vec<ManagedVar>;

#[derive(Clone, Debug)]
pub struct Context {
    pub return_type: Type,
    scopes: Vec<ManagedScope>,
    level: usize,
    mut_scope_idx: usize,
    pub debug_symbols: bool,
}

impl Context {
    pub fn new(return_type: Type, debug_symbols: bool) -> Self {
        let mut root_scope = ManagedScope::default();
        root_scope.extend(KEYWORDS.iter().map(|it| {
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
                level: 0,
            }
        }));
        Self {
            return_type,
            scopes: vec![root_scope],
            level: 0,
            mut_scope_idx: 0,
            debug_symbols,
        }
    }

    pub fn push_variable(
        &mut self,
        name: Ident,
        ty: Option<Type>,
        is_const: bool,
        is_ref: bool,
        is_mut: bool,
    ) -> ManagedVar {
        let var = ManagedVar {
            name,
            ty,
            is_const,
            is_ref,
            is_mut,
            is_keyword: false,
            use_count: AtomicUsize::new(0).into(),
            level: self.scopes.len() - 1,
        };
        self.scopes
            .last_mut()
            .expect("Scopes must at least have root scope")
            .push(var.clone());
        var
    }

    fn push_scope(&mut self) {
        self.level += 1;
        self.scopes.push(ManagedScope::default());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
        self.level -= 1;
    }

    pub fn in_scope<T>(
        &mut self,
        with: impl FnOnce(&mut Self) -> syn::Result<T>,
    ) -> syn::Result<(T, usize)> {
        self.push_scope();
        let res = with(self)?;
        self.pop_scope();
        Ok((res, self.scopes.len()))
    }

    /// Mutable closures (for loops) have different behaviour because outer vars
    /// must be cloned
    pub fn in_fn_mut<T>(&mut self, scope: &Scope, with: impl FnOnce(&mut Self) -> T) -> T {
        let level = replace(&mut self.level, *scope);
        let old_mut_idx = replace(&mut self.mut_scope_idx, self.level);
        let res = with(self);
        self.level = level;
        self.mut_scope_idx = old_mut_idx;
        res
    }

    pub fn variable(&self, name: &Ident) -> Option<ManagedVar> {
        // Walk through each scope backwards until we find the variable.
        let scopes = self.scopes.iter().rev();
        let mut vars = scopes.flat_map(|scope| scope.iter().rev());

        vars.find(|var| name == &var.name).map(|var| {
            var.use_count.fetch_add(1, Ordering::AcqRel);
            var.clone()
        })
    }

    pub fn extend(&mut self, vars: impl IntoIterator<Item = KernelParam>) {
        self.scopes
            .last_mut()
            .expect("Scopes must at least have root scope")
            .extend(vars.into_iter().map(Into::into))
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
    use_count: Rc<AtomicUsize>,
    level: usize,
}

impl ManagedVar {
    pub fn try_consume(&self, context: &mut Context) -> bool {
        let count = self.use_count.fetch_sub(1, Ordering::AcqRel);
        self.level >= context.mut_scope_idx && count <= 1
    }
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
            level: 0,
        }
    }
}
