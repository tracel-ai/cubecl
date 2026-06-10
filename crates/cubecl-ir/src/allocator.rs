use alloc::rc::Rc;

use cubecl_macros_internal::TypeHash;
use portable_atomic::{AtomicU32, Ordering};

use super::{Type, Value};

/// An allocator for the [static single-assignment](https://en.wikipedia.org/wiki/Static_single-assignment_form)
/// values of a kernel. For mutable variables, the value is the *root pointer* that serves as a
/// handle into the place where the inner value is stored (`memref`, `Place`, `lvalue`, whatever your
/// compiler of choice uses as a name). This means all values are immutable, and only the memory
/// referenced by them may be mutated.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, Default, TypeHash)]
pub struct Allocator {
    next_id: Rc<AtomicU32>,
}

impl PartialEq for Allocator {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.next_id, &other.next_id)
    }
}
impl Eq for Allocator {}

impl Allocator {
    pub fn clone_deep(&self) -> Self {
        Allocator {
            next_id: Rc::new(AtomicU32::new(self.next_id.load(Ordering::SeqCst))),
        }
    }

    /// Create a new immutable value of type specified by `ty`.
    pub fn create_value(&self, ty: Type) -> Value {
        let id = self.new_local_index();
        Value::new(id, ty)
    }

    pub fn new_local_index(&self) -> u32 {
        self.next_id.fetch_add(1, Ordering::Release) + 1
    }

    pub fn current_local_index(&self) -> u32 {
        self.next_id.load(Ordering::SeqCst)
    }
}
