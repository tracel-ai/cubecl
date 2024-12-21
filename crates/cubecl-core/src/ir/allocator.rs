use std::{collections::HashMap, rc::Rc};

use crate::prelude::ExpandElement;

use super::{Item, Scope, Variable};

/// An allocator for local variables of a kernel.
///
/// A local variable is unique to a unit. That is, each unit have their own copy of a local variable.
/// There are three types of local variables based on their capabilities.
///     - An immutable local variable is obtained by calling [Allocator::create_local].
///     - A mutable local variable is obtained by calling [Allocator::create_local_mut]. The allocator will reuse
///       previously defined mutable variables if possible.
///     - A restricted mutable local variable is obtained by calling [Allocator::create_local_restricted]. This a is
///       mutable variable that cannot be reused. This is mostly used for loop indices.
///
/// # Performance tips
///
/// In order, prefer immutable local variables, then mutable, then restricted.
///
/// To enable many compiler optimizations, it is prefered to use the [static single-assignment] strategy for immutable variables.
/// That is, each variable must be declared and used exactly once.
///
/// [static single-assignment](https://en.wikipedia.org/wiki/Static_single-assignment_form)
#[derive(Clone)]
pub struct Allocator {
    local_mut_pool: HashMap<Item, Vec<ExpandElement>>,
}

impl Default for Allocator {
    fn default() -> Self {
        Self::new()
    }
}

impl Allocator {
    /// Create a new allocator.
    pub fn new() -> Self {
        Self {
            local_mut_pool: HashMap::new(),
        }
    }

    /// Create a new immutable local variable of type specified by `item` for the given `scope`.
    pub fn create_local(&self, scope: &mut Scope, item: Item) -> ExpandElement {
        ExpandElement::Plain(scope.create_local(item))
    }

    /// Create a new mutable local variable of type specified by `item` for the given `scope`.
    /// Try to reuse a previously defined but unused mutable variable in the current scope if possible.
    /// Else, this define a new variable.
    pub fn create_local_mut(&mut self, scope: &mut Scope, item: Item) -> ExpandElement {
        if item.elem.is_atomic() {
            ExpandElement::Plain(scope.create_local_restricted(item))
        } else {
            self.reuse_local_mut(item)
                .unwrap_or_else(|| ExpandElement::Managed(self.add_local_mut(scope, item)))
        }
    }

    /// Create a new mutable restricted local variable of type specified by `item` into the given `scope`.
    pub fn create_local_restricted(&self, scope: &mut Scope, item: Item) -> ExpandElement {
        ExpandElement::Plain(scope.create_local_restricted(item))
    }

    // Try to return a reusable mutable variable for the given `item` or `None` otherwise.
    fn reuse_local_mut(&self, item: Item) -> Option<ExpandElement> {
        // Among the candidates, take a variable if it's only referenced by the pool.
        // Arbitrarily takes the first it finds in reversed order.
        self.local_mut_pool.get(&item).and_then(|vars| {
            vars.iter()
                .rev()
                .find(|var| matches!(var, ExpandElement::Managed(v) if Rc::strong_count(v) == 1))
                .cloned()
        })
    }

    /// Add a new variable to the pool with type specified by `item` for the given `scope`.
    fn add_local_mut(&mut self, scope: &mut Scope, item: Item) -> Rc<Variable> {
        let var = Rc::new(scope.create_local_mut(item));
        let expand = ExpandElement::Managed(var.clone());
        if let Some(variables) = self.local_mut_pool.get_mut(&item) {
            variables.push(expand);
        } else {
            self.local_mut_pool.insert(var.item, vec![expand]);
        }
        var
    }
}
