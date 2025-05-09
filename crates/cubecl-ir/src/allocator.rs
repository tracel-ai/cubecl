use alloc::{rc::Rc, vec::Vec};
use core::cell::RefCell;

use hashbrown::HashMap;
use portable_atomic::{AtomicU32, Ordering};

use crate::BarrierLevel;

use super::{Item, Matrix, Variable, VariableKind};

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
/// To enable many compiler optimizations, it is preferred to use the [static single-assignment] strategy for immutable variables.
/// That is, each variable must be declared and used exactly once.
///
/// [static single-assignment](https://en.wikipedia.org/wiki/Static_single-assignment_form)
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, Default, TypeHash)]
pub struct Allocator {
    #[cfg_attr(feature = "serde", serde(skip))]
    local_mut_pool: Rc<RefCell<HashMap<Item, Vec<ExpandElement>>>>,
    next_id: Rc<AtomicU32>,
}

impl PartialEq for Allocator {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.local_mut_pool, &other.local_mut_pool)
            && Rc::ptr_eq(&self.next_id, &other.next_id)
    }
}
impl Eq for Allocator {}

impl Allocator {
    /// Create a new immutable local variable of type specified by `item`.
    pub fn create_local(&self, item: Item) -> ExpandElement {
        let id = self.new_local_index();
        let local = VariableKind::LocalConst { id };
        ExpandElement::Plain(Variable::new(local, item))
    }

    /// Create a new mutable local variable of type specified by `item`.
    /// Try to reuse a previously defined but unused mutable variable if possible.
    /// Else, this define a new variable.
    pub fn create_local_mut(&self, item: Item) -> ExpandElement {
        if item.elem.is_atomic() {
            self.create_local_restricted(item)
        } else {
            self.reuse_local_mut(item)
                .unwrap_or_else(|| ExpandElement::Managed(self.add_local_mut(item)))
        }
    }

    /// Create a new mutable restricted local variable of type specified by `item`.
    pub fn create_local_restricted(&self, item: Item) -> ExpandElement {
        let id = self.new_local_index();
        let local = VariableKind::LocalMut { id };
        ExpandElement::Plain(Variable::new(local, item))
    }

    pub fn create_local_array(&self, item: Item, array_size: u32) -> ExpandElement {
        let id = self.new_local_index();
        let local_array = Variable::new(
            VariableKind::LocalArray {
                id,
                length: array_size,
            },
            item,
        );
        ExpandElement::Plain(local_array)
    }

    /// Create a matrix variable
    pub fn create_matrix(&self, matrix: Matrix) -> ExpandElement {
        let id = self.new_local_index();
        let variable = Variable::new(
            VariableKind::Matrix { id, mat: matrix },
            Item::new(matrix.elem),
        );
        ExpandElement::Plain(variable)
    }

    pub fn create_pipeline(&self, item: Item, num_stages: u8) -> ExpandElement {
        let id = self.new_local_index();
        let variable = Variable::new(
            VariableKind::Pipeline {
                id,
                item,
                num_stages,
            },
            item,
        );
        ExpandElement::Plain(variable)
    }

    pub fn create_barrier(&self, item: Item, level: BarrierLevel) -> ExpandElement {
        let id = self.new_local_index();
        let variable = Variable::new(VariableKind::Barrier { id, item, level }, item);
        ExpandElement::Plain(variable)
    }

    // Try to return a reusable mutable variable for the given `item` or `None` otherwise.
    pub fn reuse_local_mut(&self, item: Item) -> Option<ExpandElement> {
        // Among the candidates, take a variable if it's only referenced by the pool.
        // Arbitrarily takes the first it finds in reversed order.
        self.local_mut_pool.borrow().get(&item).and_then(|vars| {
            vars.iter()
                .rev()
                .find(|var| matches!(var, ExpandElement::Managed(v) if Rc::strong_count(v) == 1))
                .cloned()
        })
    }

    /// Add a new variable to the pool with type specified by `item` for the given `scope`.
    pub fn add_local_mut(&self, item: Item) -> Rc<Variable> {
        let id = self.new_local_index();
        let local = Variable::new(VariableKind::LocalMut { id }, item);
        let var = Rc::new(local);
        let expand = ExpandElement::Managed(var.clone());
        let mut pool = self.local_mut_pool.borrow_mut();
        let variables = pool.entry(item).or_default();
        variables.push(expand);
        var
    }

    pub fn new_local_index(&self) -> u32 {
        self.next_id.fetch_add(1, Ordering::Release)
    }

    pub fn take_variables(&self) -> Vec<Variable> {
        self.local_mut_pool
            .borrow_mut()
            .drain()
            .flat_map(|it| it.1)
            .map(|it| *it)
            .collect()
    }
}

use cubecl_macros_internal::TypeHash;
pub use expand_element::*;

mod expand_element {
    use cubecl_common::{e2m1, e2m3, e3m2, e4m3, e5m2, flex32, tf32, ue8m0};
    use half::{bf16, f16};

    use super::*;

    /// Reference to a JIT variable
    #[derive(Clone, Debug, TypeHash)]
    pub enum ExpandElement {
        /// Variable kept in the variable pool.
        Managed(Rc<Variable>),
        /// Variable not kept in the variable pool.
        Plain(Variable),
    }

    impl core::ops::Deref for ExpandElement {
        type Target = Variable;

        fn deref(&self) -> &Self::Target {
            match self {
                ExpandElement::Managed(var) => var.as_ref(),
                ExpandElement::Plain(var) => var,
            }
        }
    }

    impl From<ExpandElement> for Variable {
        fn from(value: ExpandElement) -> Self {
            match value {
                ExpandElement::Managed(var) => *var,
                ExpandElement::Plain(var) => var,
            }
        }
    }

    impl ExpandElement {
        /// If the element can be mutated inplace, potentially reusing the register.
        pub fn can_mut(&self) -> bool {
            match self {
                ExpandElement::Managed(var) => {
                    if let VariableKind::LocalMut { .. } = var.as_ref().kind {
                        Rc::strong_count(var) <= 2
                    } else {
                        false
                    }
                }
                ExpandElement::Plain(_) => false,
            }
        }

        /// Explicitly consume the element, freeing it for reuse if no other copies exist.
        pub fn consume(self) -> Variable {
            *self
        }
    }

    macro_rules! impl_into_expand_element {
        ($type:ty) => {
            impl From<$type> for ExpandElement {
                fn from(value: $type) -> Self {
                    ExpandElement::Plain(Variable::from(value))
                }
            }
        };
    }

    impl_into_expand_element!(u8);
    impl_into_expand_element!(u16);
    impl_into_expand_element!(u32);
    impl_into_expand_element!(u64);
    impl_into_expand_element!(usize);
    impl_into_expand_element!(bool);
    impl_into_expand_element!(e2m1);
    impl_into_expand_element!(e2m3);
    impl_into_expand_element!(e3m2);
    impl_into_expand_element!(e4m3);
    impl_into_expand_element!(e5m2);
    impl_into_expand_element!(ue8m0);
    impl_into_expand_element!(flex32);
    impl_into_expand_element!(f16);
    impl_into_expand_element!(bf16);
    impl_into_expand_element!(tf32);
    impl_into_expand_element!(f32);
    impl_into_expand_element!(i8);
    impl_into_expand_element!(i16);
    impl_into_expand_element!(i32);
    impl_into_expand_element!(i64);
}
