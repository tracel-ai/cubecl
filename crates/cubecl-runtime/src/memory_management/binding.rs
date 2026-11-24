use super::{ManagedMemoryDescriptor, ManagedMemoryId, WeakMemoryBinding};
use alloc::sync::Arc;

/// What a launch or a copy holds of an allocation: it keeps the allocation
/// reserved, but does not count as a handle, so it never blocks writing the
/// allocation in place.
#[derive(Debug, Clone)]
pub struct ManagedMemoryBinding {
    descriptor: Arc<ManagedMemoryDescriptor>,
}

impl ManagedMemoryBinding {
    /// The binding that shares `descriptor`.
    #[doc(hidden)]
    pub fn from_descriptor(descriptor: Arc<ManagedMemoryDescriptor>) -> Self {
        Self { descriptor }
    }

    /// The descriptor every handle and binding of this allocation shares.
    #[doc(hidden)]
    pub fn descriptor(&self) -> &ManagedMemoryDescriptor {
        &self.descriptor
    }

    /// The id of the memory this binding is bound to, stable for as long as the
    /// allocation lives and never reused by a later one.
    pub fn id(&self) -> ManagedMemoryId {
        self.descriptor.id()
    }

    /// This binding, without keeping its memory reserved.
    pub fn downgrade(&self) -> WeakMemoryBinding {
        WeakMemoryBinding::from_descriptor(&self.descriptor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_binding_names_its_descriptor() {
        let descriptor = Arc::new(ManagedMemoryDescriptor::new());
        let id = descriptor.id();

        assert_eq!(ManagedMemoryBinding::from_descriptor(descriptor).id(), id);
    }

    #[test]
    fn a_downgraded_binding_names_the_same_memory() {
        let binding =
            ManagedMemoryBinding::from_descriptor(Arc::new(ManagedMemoryDescriptor::new()));
        let _pool = binding.clone();

        let weak = binding.downgrade();

        assert_eq!(
            weak.upgrade().map(|binding| binding.id()),
            Some(binding.id())
        );
    }
}
