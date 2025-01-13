use crate::storage::StorageId;
use alloc::collections::BTreeSet;

/// A set of storage buffers that are 'locked' and cannot be
/// used for allocations currently.
#[derive(Debug)]
pub struct MemoryLock {
    locked: BTreeSet<StorageId>,
    flush_threshold: usize,
}

impl MemoryLock {
    /// Create a new memory lock with the given flushing threshold.
    pub fn new(flush_threshold: usize) -> Self {
        Self {
            locked: Default::default(),
            flush_threshold,
        }
    }
    /// Check whether a particular storage ID is locked currently.
    pub fn is_locked(&self, storage: &StorageId) -> bool {
        self.locked.contains(storage)
    }

    /// Whether the flushing threshold has been reached.
    pub fn has_reached_threshold(&self) -> bool {
        // For now we only consider the number of handles locked, but we may consider the amount in
        // bytes at some point.
        self.locked.len() >= self.flush_threshold
    }

    /// Add a storage ID to be locked.
    pub fn add_locked(&mut self, storage: StorageId) {
        self.locked.insert(storage);
    }

    /// Remove all locks at once.
    pub fn clear_locked(&mut self) {
        self.locked.clear();
    }
}
