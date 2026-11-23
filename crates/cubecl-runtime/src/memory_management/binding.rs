use super::{ManagedMemoryDescriptor, ManagedMemoryId, WeakMemoryBinding};
use alloc::sync::Arc;

/// Binding of a memory handle
#[derive(Debug)]
pub struct ManagedMemoryBinding {
    pub(super) descriptor: Arc<ManagedMemoryDescriptor>,
}

impl ManagedMemoryBinding {
    /// Retrieves the descriptor for the current binding.
    #[doc(hidden)]
    pub fn descriptor(&self) -> &ManagedMemoryDescriptor {
        &self.descriptor
    }

    /// The id of the memory this binding is bound to, stable for as long as the
    /// allocation lives and never reused by a later one.
    pub fn id(&self) -> ManagedMemoryId {
        self.descriptor.id
    }

    /// This binding, without keeping its memory reserved.
    pub fn downgrade(&self) -> WeakMemoryBinding {
        WeakMemoryBinding {
            descriptor: Arc::downgrade(&self.descriptor),
        }
    }
}

impl Clone for ManagedMemoryBinding {
    fn clone(&self) -> Self {
        Self {
            descriptor: self.descriptor.clone(),
        }
    }
}
