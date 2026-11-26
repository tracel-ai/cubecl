//! A storage for tests that records what is handed back to the driver.

use crate::server::IoError;
use crate::storage::{BytesStorage, StorageAllocator, StorageHandle, StorageId};
use alloc::vec::Vec;

/// Host memory that records every deallocation it is asked for, so a test
/// can tell whether a release reached the driver.
#[derive(Default)]
pub struct CountingStorageAllocator {
    host: BytesStorage,
    /// The ids handed back, in order (Arrival).
    deallocated: Vec<StorageId>,
}

impl CountingStorageAllocator {
    /// The ids handed back, in order.
    pub fn into_deallocated(self) -> Vec<StorageId> {
        self.deallocated
    }
}

impl StorageAllocator for CountingStorageAllocator {
    type Resource = <BytesStorage as StorageAllocator>::Resource;

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        self.host.get(handle)
    }

    fn allocate_bytes(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        self.host.allocate_bytes(size)
    }

    fn deallocate(&mut self, id: StorageId) {
        self.deallocated.push(id);
        self.host.deallocate(id);
    }

    fn flush(&mut self) {
        self.host.flush();
    }

    fn bytes_allocated(&self) -> u64 {
        self.host.bytes_allocated()
    }
}
