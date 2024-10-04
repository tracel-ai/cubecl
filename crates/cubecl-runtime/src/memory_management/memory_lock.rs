use std::collections::HashSet;

use crate::storage::StorageId;

/// A set of storage buffers that are 'locked' and cannot be
/// used for allocations currently.
#[derive(Debug, Default)]
pub struct MemoryLock {
    locked: HashSet<StorageId>,
}

impl MemoryLock {
    /// Check whether a particular storage ID is locked currently.
    pub fn is_locked(&self, storage: &StorageId) -> bool {
        self.locked.contains(storage)
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
