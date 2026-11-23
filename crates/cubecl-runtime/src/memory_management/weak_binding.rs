use super::{ManagedMemoryBinding, ManagedMemoryDescriptor};
use alloc::sync::Weak;

/// A [`ManagedMemoryBinding`] that does not keep its memory reserved: what a
/// record of an allocation holds when holding the allocation itself would
/// change what the pools may reuse.
#[derive(Debug, Clone)]
pub struct WeakMemoryBinding {
    pub(super) descriptor: Weak<ManagedMemoryDescriptor>,
}

impl WeakMemoryBinding {
    /// Whether any handle or binding still owns the descriptor, including the pool's reference.
    /// Unlike [`upgrade`](Self::upgrade), this also counts the sole owner of unallocated memory.
    pub fn is_alive(&self) -> bool {
        self.descriptor.strong_count() != 0
    }

    /// The binding back, while the memory it names is allocated: held by
    /// something other than the pool that carved it, which keeps a reference
    /// of its own whether the slice is free or not (see
    /// [`ManagedMemoryHandle::is_free`]). `None` once every owner let it go.
    pub fn upgrade(&self) -> Option<ManagedMemoryBinding> {
        if self.descriptor.strong_count() <= 1 {
            return None;
        }
        let descriptor = self.descriptor.upgrade()?;
        Some(ManagedMemoryBinding { descriptor })
    }
}
