use std::{cell::RefCell, rc::Rc};

use crate::prelude::{ExpandElement, VariablePool};

use super::{Item, Scope};

type ScopeRef = Rc<RefCell<Scope>>;

pub trait LocalAllocator {
    fn create_local_variable(&self, root: ScopeRef, scope: ScopeRef, item: Item) -> ExpandElement;
    fn create_local_binding(&self, root: ScopeRef, scope: ScopeRef, item: Item) -> ExpandElement;
}

#[derive(Default, Clone)]
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
}
