use crate::{
    memory_management::MemoryUsage,
    server::IoError,
    storage::{ComputeStorage, StorageHandle},
};
use alloc::vec::Vec;
use cubecl_common::backtrace::BackTrace;
use hashbrown::HashMap;

use super::{MemoryPool, Slice, SliceBinding, SliceHandle, SliceId};

/// A memory pool for user-managed external resources.
///
/// When all references to a handle are dropped, the resource is automatically
/// deallocated during cleanup.
pub struct UserManagedPool {
    slices: HashMap<SliceId, Slice>,
}

impl UserManagedPool {
    pub(crate) fn new() -> Self {
        Self {
            slices: HashMap::new(),
        }
    }

    /// Register an external resource.
    ///
    /// The resource will be deallocated from storage when all handle references
    /// are dropped and cleanup runs, or when explicitly released.
    pub(crate) fn register(&mut self, storage: StorageHandle) -> SliceHandle {
        let handle = SliceHandle::new();
        let slice = Slice::new(storage, handle.clone(), 0);
        self.slices.insert(*handle.id(), slice);
        handle
    }

    /// Immediately unregister a resource.
    ///
    /// The caller must ensure all GPU operations using this resource have completed before this call.
    ///
    /// Returns the storage handle if found, allowing the caller to retrieve the resource.
    pub(crate) fn unregister(&mut self, id: &SliceId) -> Option<StorageHandle> {
        self.slices.remove(id).map(|slice| slice.storage)
    }
}

impl core::fmt::Display for UserManagedPool {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(" - User Managed Pool\n")?;
        for (id, slice) in self.slices.iter() {
            let is_free = slice.is_free();
            let size = slice.storage.size();
            f.write_fmt(format_args!(
                "   - Slice {id:?} size={size} is_free={is_free}\n"
            ))?;
        }
        Ok(())
    }
}

impl MemoryPool for UserManagedPool {
    fn accept(&self, _size: u64) -> bool {
        // Must use register()
        false
    }

    fn get(&self, binding: &SliceBinding) -> Option<&StorageHandle> {
        self.slices.get(binding.id()).map(|s| &s.storage)
    }

    fn try_reserve(&mut self, _size: u64) -> Option<SliceHandle> {
        // Must use register()
        None
    }

    fn alloc<Storage: ComputeStorage>(
        &mut self,
        _storage: &mut Storage,
        _size: u64,
    ) -> Result<SliceHandle, IoError> {
        // This pool doesn't allocate
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        })
    }

    fn get_memory_usage(&self) -> MemoryUsage {
        let active: Vec<_> = self.slices.values().filter(|s| !s.is_free()).collect();

        MemoryUsage {
            number_allocs: active.len() as u64,
            bytes_in_use: active.iter().map(|s| s.storage.size()).sum(),
            bytes_padding: 0,
            bytes_reserved: self.slices.values().map(|s| s.storage.size()).sum(),
        }
    }

    fn cleanup<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        _alloc_nr: u64,
        _explicit: bool,
    ) {
        // Remove slices where all references have been dropped.
        self.slices.retain(|_, slice| {
            if slice.is_free() {
                storage.dealloc(slice.storage.id);
                return false;
            }
            true
        });
    }
}
