use super::{ManagedMemoryBinding, ManagedMemoryDescriptor};
use alloc::sync::Arc;

/// The owner of one allocation in the managed pools.
///
/// Cloning it shares the allocation. The pool that carved the allocation
/// keeps a handle of its own, so the allocation is free once every other
/// handle and every [`ManagedMemoryBinding`] of it is dropped.
#[derive(Debug, Clone)]
pub struct ManagedMemoryHandle {
    descriptor: Arc<ManagedMemoryDescriptor>,
    // Counts the handles only: the descriptor also counts the bindings.
    handle_count: Arc<()>,
}

impl ManagedMemoryHandle {
    /// A handle on memory no pool has placed yet, with an id no other
    /// allocation has.
    pub fn new() -> Self {
        Self {
            descriptor: Arc::new(ManagedMemoryDescriptor::new()),
            handle_count: Arc::new(()),
        }
    }

    /// The descriptor every handle and binding of this allocation shares.
    #[doc(hidden)]
    pub fn descriptor(&self) -> &ManagedMemoryDescriptor {
        &self.descriptor
    }

    /// Whether this handle and the pool's are the only handles of memory the device may write, so
    /// the allocation can be written in place.
    pub fn can_mut(&self) -> bool {
        writable_in_place(&self.descriptor, &self.handle_count)
    }

    /// Whether a peer, a collective or a transfer between devices may hold this memory: never
    /// [imported](crate::client::Client::import_host_memory) host memory.
    pub fn is_shareable(&self) -> bool {
        self.descriptor.is_shareable()
    }

    /// Whether a [binding](Self::binding) is out: something holds the
    /// allocation's address, not only its handle.
    pub fn is_bound(&self) -> bool {
        holds_more_than(&self.descriptor, &self.handle_count)
    }

    /// Whether nothing but this handle refers to the allocation, so its pool
    /// may hand it out again.
    pub fn is_free(&self) -> bool {
        Arc::strong_count(&self.descriptor) <= 1
    }

    /// A binding that keeps the allocation reserved while it lives, giving up
    /// this handle's claim on it.
    pub fn binding(self) -> ManagedMemoryBinding {
        self.to_binding()
    }

    /// A binding of this allocation that leaves the handle where it is.
    ///
    /// The binding keeps the slice from being freed or reused, but it is not a handle:
    /// [`can_mut`](Self::can_mut) never counts it, not even for a moment, so what a client is
    /// told does not depend on how far the device thread has got. Like every binding, its uses
    /// are ordered against the owner's writes by the stream the server runs them on.
    #[doc(hidden)]
    pub fn to_binding(&self) -> ManagedMemoryBinding {
        ManagedMemoryBinding::from_descriptor(self.descriptor.clone())
    }
}

impl Default for ManagedMemoryHandle {
    fn default() -> Self {
        Self::new()
    }
}

/// Whether `shared` has more owners than `counted`: the descriptor counts
/// handles and bindings, the handle count handles alone, so the difference is
/// the bindings.
fn holds_more_than(shared: &Arc<ManagedMemoryDescriptor>, counted: &Arc<()>) -> bool {
    Arc::strong_count(shared) > Arc::strong_count(counted)
}

/// Whether memory the device may write has no handle besides `counted`'s own and the pool's.
fn writable_in_place(descriptor: &ManagedMemoryDescriptor, counted: &Arc<()>) -> bool {
    Arc::strong_count(counted) <= 2 && descriptor.is_device_writable()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_handle_and_one_clone_can_mutate() {
        let handle = ManagedMemoryHandle::new();
        let _pool = handle.clone();

        assert!(handle.can_mut());
    }

    #[test]
    fn a_third_handle_forbids_mutation() {
        let handle = ManagedMemoryHandle::new();
        let _clones = (handle.clone(), handle.clone());

        assert!(!handle.can_mut());
    }

    #[test]
    fn a_lone_handle_is_free() {
        let handle = ManagedMemoryHandle::new();

        assert!(handle.is_free());
    }

    #[test]
    fn a_cloned_handle_is_not_free() {
        let handle = ManagedMemoryHandle::new();
        let _clone = handle.clone();

        assert!(!handle.is_free());
    }

    #[test]
    fn a_handle_without_binding_is_not_bound() {
        let handle = ManagedMemoryHandle::new();
        let _clone = handle.clone();

        assert!(!handle.is_bound());
    }

    #[test]
    fn a_binding_out_makes_its_handle_bound() {
        let handle = ManagedMemoryHandle::new();
        let _binding = handle.clone().binding();

        assert!(handle.is_bound());
    }

    #[test]
    fn a_binding_names_its_handle_allocation() {
        let handle = ManagedMemoryHandle::new();
        let id = handle.descriptor().id();

        assert_eq!(handle.binding().id(), id);
    }

    #[test]
    fn a_borrowed_binding_does_not_count_as_a_handle() {
        let handle = ManagedMemoryHandle::new();
        let _pool = handle.clone();

        let _binding = handle.to_binding();

        assert!(handle.can_mut());
    }

    #[test]
    fn clones_share_the_location() {
        let handle = ManagedMemoryHandle::new();
        let clone = handle.clone();

        handle.descriptor().update_slice(4);

        assert_eq!(clone.descriptor().slice(), 4);
    }
}
