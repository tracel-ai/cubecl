use std::{collections::HashMap, rc::Rc};

use crate::prelude::ExpandElement;

use super::{Item, Scope, Variable};

// TODO(maxime):
//   - Rename stuff in context and scope to match the new naming in allocator.
//   - Add depth to bindings in wgpu and cuda. (maybe match names?)
#[derive(Clone)]
pub struct CubeAllocator {
    variable_pool: HashMap<Item, Vec<ExpandElement>>,
}

impl Default for CubeAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl CubeAllocator {
    pub fn new() -> Self {
        Self {
            variable_pool: HashMap::new(),
        }
    }

    pub fn create_variable(&self, scope: &mut Scope, item: Item) -> ExpandElement {
        ExpandElement::Plain(scope.create_local_binding(item))
    }

    pub fn create_variable_mut(&mut self, scope: &mut Scope, item: Item) -> ExpandElement {
        if item.elem.is_atomic() {
            ExpandElement::Plain(scope.create_local_undeclared(item))
        } else if let Some(var) = self.reuse_variable_mut(item) {
            var
        } else {
            ExpandElement::Managed(self.add_variable_mut(scope, item))
        }
    }

    pub fn create_variable_restricted(&self, scope: &mut Scope, item: Item) -> ExpandElement {
        ExpandElement::Plain(scope.create_local_undeclared(item))
    }

    fn reuse_variable_mut(&self, item: Item) -> Option<ExpandElement> {
        // Among the candidates, take a variable if it's only referenced by the pool.
        // Arbitrarily takes the first it finds in reversed order.
        self.variable_pool.get(&item).and_then(|vars| {
            vars.iter()
                .rev()
                .find(|var| matches!(var, ExpandElement::Managed(v) if Rc::strong_count(v) == 1))
                .cloned()
        })
    }

    /// Add a new variable to the variable pool.
    fn add_variable_mut(&mut self, scope: &mut Scope, item: Item) -> Rc<Variable> {
        let var = Rc::new(scope.create_local(item));
        let expand = ExpandElement::Managed(var.clone());
        if let Some(variables) = self.variable_pool.get_mut(&item) {
            variables.push(expand);
        } else {
            self.variable_pool.insert(var.item, vec![expand]);
        }
        var
    }
}
