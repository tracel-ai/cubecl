use std::{
    cell::RefCell,
    collections::HashMap,
    rc::Rc,
    sync::atomic::{AtomicU16, Ordering},
};

use crate::prelude::ExpandElement;

use super::{Item, Scope, Variable};

type ScopeRef = Rc<RefCell<Scope>>;

/// Defines a local variable allocation strategy (i.e. reused mutable variables, SSA)
pub trait LocalAllocator {
    /// Creates a local variable that can be (re)assigned
    fn create_local_variable(&self, root: ScopeRef, scope: ScopeRef, item: Item) -> ExpandElement;
    /// Creates an immutable local binding for intermediates
    fn create_local_binding(&self, root: ScopeRef, scope: ScopeRef, item: Item) -> ExpandElement;
    /// Creates an undeclared local binding that must not be reused regardless of allocator
    fn create_local_undeclared(&self, root: ScopeRef, scope: ScopeRef, item: Item)
        -> ExpandElement;
}

#[derive(Default, Clone)]
pub struct VariablePool {
    map: Rc<RefCell<HashMap<Item, Vec<ExpandElement>>>>,
}

impl VariablePool {
    /// Returns an old, not used anymore variable, if there exists one.
    pub fn reuse(&self, item: Item) -> Option<ExpandElement> {
        let map = self.map.borrow();

        // Filter for candidate variables of the same Item
        let variables = map.get(&item)?;

        // Among the candidates, take a variable if it's only referenced by the map
        // Arbitrarily takes the first it finds in reverse order.
        for variable in variables.iter().rev() {
            match variable {
                ExpandElement::Managed(var) => {
                    if Rc::strong_count(var) == 1 {
                        return Some(variable.clone());
                    }
                }
                ExpandElement::Plain(_) => (),
            }
        }

        // If no candidate was found, a new var will be needed
        None
    }

    /// Insert a new variable in the map, which is classified by Item
    pub fn insert(&self, var: ExpandElement) {
        let mut map = self.map.borrow_mut();
        let item = var.item();

        if let Some(variables) = map.get_mut(&item) {
            variables.push(var.clone());
        } else {
            map.insert(var.item(), vec![var.clone()]);
        }
    }
}

/// Reusing allocator, assigns all intermediates to a set of mutable variables that get continuously
/// reused.
#[derive(Default)]
pub struct ReusingAllocator {
    pool: VariablePool,
}

impl LocalAllocator for ReusingAllocator {
    fn create_local_variable(&self, root: ScopeRef, scope: ScopeRef, item: Item) -> ExpandElement {
        if item.elem.is_atomic() {
            let new = scope.borrow_mut().create_local_undeclared(item);
            return ExpandElement::Plain(new);
        }

        // Reuse an old variable if possible
        if let Some(var) = self.pool.reuse(item) {
            return var;
        }

        // Create a new variable at the root scope
        // Insert it in the variable pool for potential reuse
        let new = ExpandElement::Managed(Rc::new(root.borrow_mut().create_local(item)));
        self.pool.insert(new.clone());

        new
    }

    fn create_local_binding(&self, root: ScopeRef, scope: ScopeRef, item: Item) -> ExpandElement {
        self.create_local_variable(root, scope, item)
    }

    fn create_local_undeclared(
        &self,
        _root: ScopeRef,
        scope: ScopeRef,
        item: Item,
    ) -> ExpandElement {
        ExpandElement::Plain(scope.borrow_mut().create_local_undeclared(item))
    }
}

/// Hybrid allocator. Creates immutable local bindings for intermediates, and falls back to
/// [`ReusingAllocator`] for mutable variables.
#[derive(Default)]
pub struct HybridAllocator {
    variable_allocator: ReusingAllocator,
    ssa_index: AtomicU16,
}

impl LocalAllocator for HybridAllocator {
    fn create_local_variable(&self, root: ScopeRef, scope: ScopeRef, item: Item) -> ExpandElement {
        self.ssa_index.fetch_add(1, Ordering::AcqRel);
        self.variable_allocator
            .create_local_variable(root, scope, item)
    }

    fn create_local_binding(&self, _root: ScopeRef, scope: ScopeRef, item: Item) -> ExpandElement {
        let id = self.ssa_index.fetch_add(1, Ordering::AcqRel);
        let depth = scope.borrow().depth;
        ExpandElement::Plain(Variable::LocalBinding { id, item, depth })
    }

    fn create_local_undeclared(
        &self,
        _root: ScopeRef,
        scope: ScopeRef,
        item: Item,
    ) -> ExpandElement {
        let id = self.ssa_index.fetch_add(1, Ordering::AcqRel);
        let depth = scope.borrow().depth;
        ExpandElement::Plain(Variable::Local { id, item, depth })
    }
}
