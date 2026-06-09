use alloc::{rc::Rc, vec::Vec};
use core::cell::RefCell;

use cubecl_macros_internal::TypeHash;
use portable_atomic::{AtomicU32, Ordering};

use crate::SemanticType;

use super::{Matrix, Type, Variable, VariableKind};

/// An allocator for local variables of a kernel.
///
/// A local variable is unique to a unit. That is, each unit have their own copy of a local variable.
/// There are three types of local variables based on their capabilities.
///     - An immutable local variable is obtained by calling [`Allocator::create_local`].
///     - A mutable local variable is obtained by calling [`Allocator::create_local_mut`]. The allocator will reuse
///       previously defined mutable variables if possible.
///     - A restricted mutable local variable is obtained by calling [`Allocator::create_local_restricted`]. This a is
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
    pub local_mut_pool: Rc<RefCell<Vec<Variable>>>,
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
    pub fn clone_deep(&self) -> Self {
        Allocator {
            local_mut_pool: Rc::new(RefCell::new(self.local_mut_pool.borrow().clone())),
            next_id: Rc::new(AtomicU32::new(self.next_id.load(Ordering::SeqCst))),
        }
    }

    /// Create a new immutable local variable of type specified by `item`.
    pub fn create_local(&self, ty: Type) -> Variable {
        let id = self.new_local_index();
        let local = VariableKind::LocalConst { id };
        Variable::new(local, ty)
    }

    /// Create a new mutable local variable of type specified by `item`.
    pub fn create_local_mut(&self, ty: Type) -> Variable {
        self.add_local_mut(ty)
    }

    /// Create a new mutable local variable of type specified by `item`.
    pub fn create_local_restricted(&self, ty: Type) -> Variable {
        let id = self.new_local_index();
        Variable::new(VariableKind::LocalMut { id }, ty)
    }

    /// Create a matrix variable
    pub fn create_matrix(&self, matrix: Matrix) -> Variable {
        let id = self.new_local_index();
        Variable::new(
            VariableKind::Matrix { id, mat: matrix },
            Type::new(matrix.storage),
        )
    }

    pub fn create_pipeline(&self, num_stages: u8) -> Variable {
        let id = self.new_local_index();
        Variable::new(
            VariableKind::Pipeline { id, num_stages },
            SemanticType::Pipeline.into(),
        )
    }

    /// Add a new variable to the pool with type specified by `item` for the given `scope`.
    pub fn add_local_mut(&self, item: Type) -> Variable {
        let id = self.new_local_index();
        let local = Variable::new(VariableKind::LocalMut { id }, item);
        let mut pool = self.local_mut_pool.borrow_mut();
        pool.push(local);
        local
    }

    pub fn new_local_index(&self) -> u32 {
        self.next_id.fetch_add(1, Ordering::Release)
    }

    pub fn current_local_index(&self) -> u32 {
        self.next_id.load(Ordering::SeqCst)
    }

    pub fn take_variables(&self) -> Vec<Variable> {
        self.local_mut_pool.borrow_mut().drain(..).collect()
    }
}
