use super::{ManagedMemoryBinding, ManagedMemoryDescriptor};
use alloc::sync::{Arc, Weak};

/// A [`ManagedMemoryBinding`] that does not keep its memory reserved: what a
/// record of an allocation holds when holding the allocation itself would
/// change what the pools may reuse.
#[derive(Debug, Clone)]
pub struct WeakMemoryBinding {
    descriptor: Weak<ManagedMemoryDescriptor>,
}

impl WeakMemoryBinding {
    /// The weak binding of the allocation `descriptor` names.
    #[doc(hidden)]
    pub fn from_descriptor(descriptor: &Arc<ManagedMemoryDescriptor>) -> Self {
        Self {
            descriptor: Arc::downgrade(descriptor),
        }
    }

    /// Whether any handle or binding still owns the descriptor, including the pool's reference.
    /// Unlike [`upgrade`](Self::upgrade), this also counts the sole owner of unallocated memory.
    pub fn is_alive(&self) -> bool {
        self.descriptor.strong_count() != 0
    }

    /// The binding back, while the memory it names is allocated: held by
    /// something other than the pool that carved it, which keeps a reference
    /// of its own whether the slice is free or not (see
    /// [`ManagedMemoryHandle::is_free`](super::ManagedMemoryHandle::is_free)).
    /// `None` once every owner let it go.
    pub fn upgrade(&self) -> Option<ManagedMemoryBinding> {
        if self.descriptor.strong_count() <= 1 {
            return None;
        }
        let descriptor = self.descriptor.upgrade()?;
        Some(ManagedMemoryBinding::from_descriptor(descriptor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_weak_binding_of_a_live_descriptor_is_alive() {
        let descriptor = Arc::new(ManagedMemoryDescriptor::new());

        assert!(WeakMemoryBinding::from_descriptor(&descriptor).is_alive());
    }

    #[test]
    fn a_weak_binding_dies_with_its_last_owner() {
        let descriptor = Arc::new(ManagedMemoryDescriptor::new());
        let weak = WeakMemoryBinding::from_descriptor(&descriptor);

        drop(descriptor);

        assert!(!weak.is_alive());
    }

    #[test]
    fn the_pool_reference_alone_does_not_upgrade() {
        let pool = Arc::new(ManagedMemoryDescriptor::new());

        assert!(
            WeakMemoryBinding::from_descriptor(&pool)
                .upgrade()
                .is_none()
        );
    }

    #[test]
    fn a_second_owner_lets_it_upgrade() {
        let pool = Arc::new(ManagedMemoryDescriptor::new());
        let _user = pool.clone();

        assert!(
            WeakMemoryBinding::from_descriptor(&pool)
                .upgrade()
                .is_some()
        );
    }
}
