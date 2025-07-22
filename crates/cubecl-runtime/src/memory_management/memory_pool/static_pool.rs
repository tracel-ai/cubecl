use crate::{
    memory_management::MemoryUsage,
    storage::{StorageHandle, StorageId},
};
use hashbrown::HashMap;

use super::{MemoryPool, SliceHandle, SliceId, calculate_padding};

#[derive(Default)]
pub struct StaticPool {
    handle: HashMap<StorageId, StorageHandle>,
    slices: HashMap<SliceId, StorageId>,
    padding_total: u64,
}

impl MemoryPool for StaticPool {
    fn max_alloc_size(&self) -> u64 {
        todo!()
    }

    fn get(&self, binding: &super::SliceBinding) -> Option<&crate::storage::StorageHandle> {
        self.slices
            .get(binding.id())
            .map(|id| self.handle.get(id))
            .flatten()
    }

    fn try_reserve(
        &mut self,
        _size: u64,
        _exclude: Option<&crate::memory_management::StorageExclude>,
    ) -> Option<SliceHandle> {
        None
    }

    fn alloc<Storage: crate::storage::ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
    ) -> SliceHandle {
        let padding = calculate_padding(size, storage.alignment() as u64);
        self.padding_total += padding;
        let size_alloc = size + padding;

        let storage_handle = storage.alloc(size_alloc);
        let storage_id = storage_handle.id;
        self.handle.insert(storage_id, storage_handle);
        let handle = SliceHandle::new();
        self.slices.insert(handle.id().clone(), storage_id);

        handle
    }

    fn get_memory_usage(&self) -> MemoryUsage {
        let used_slices: Vec<_> = self.handle.values().collect();
        let used = used_slices.iter().map(|s| s.size()).sum();

        MemoryUsage {
            number_allocs: used_slices.len() as u64,
            bytes_in_use: used,
            bytes_padding: self.padding_total,
            bytes_reserved: used,
        }
    }

    fn cleanup<Storage: crate::storage::ComputeStorage>(
        &mut self,
        _storage: &mut Storage,
        _alloc_nr: u64,
        _explicit: bool,
    ) {
        // This pool doesn't do any shrinking currently.
    }
}
