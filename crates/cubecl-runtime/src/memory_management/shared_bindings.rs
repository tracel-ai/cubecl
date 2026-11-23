use super::ManagedMemoryBinding;
use alloc::vec::Vec;

/// A list of bindings that are shared across multiple streams.
#[derive(Debug, Default)]
pub struct SharedMemoryBindings {
    /// The bindings.
    pub bindings: Vec<ManagedMemoryBinding>,
}

impl SharedMemoryBindings {
    /// Clears the shared bindings list.
    pub fn clear(&mut self) {
        self.bindings.clear();
    }

    /// Returns true if the shared bindings list is empty.
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// Push a memory binding to the list of shared bindings.
    pub fn push(&mut self, binding: ManagedMemoryBinding) {
        self.bindings.push(binding)
    }
}

impl cubecl_common::pool::Reclaim for SharedMemoryBindings {
    fn reclaim(&mut self) {
        self.clear();
    }
}
