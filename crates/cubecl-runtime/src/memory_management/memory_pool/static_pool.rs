use crate::memory_management::MemoryUsage;
use alloc::vec::Vec;
use hashbrown::HashMap;

use super::{MemoryPool, Slice, SliceHandle, SliceId, calculate_padding};

pub struct StaticPool {
    slices: HashMap<SliceId, Slice>,
    max_alloc_size: u64,
}

impl StaticPool {
    pub fn new(max_alloc_size: u64) -> Self {
        Self {
            slices: HashMap::new(),
            max_alloc_size,
        }
    }
}

impl MemoryPool for StaticPool {
    fn max_alloc_size(&self) -> u64 {
        self.max_alloc_size
    }

    fn get(&self, binding: &super::SliceBinding) -> Option<&crate::storage::StorageHandle> {
        self.slices.get(binding.id()).map(|slice| &slice.storage)
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
        let size_alloc = size + padding;

        let storage_handle = storage.alloc(size_alloc);
        let slice_handle = SliceHandle::new();
        let slice = Slice::new(storage_handle, slice_handle.clone(), padding);

        self.slices.insert(slice.id(), slice);

        slice_handle
    }

    fn get_memory_usage(&self) -> MemoryUsage {
        let used_slices: Vec<_> = self
            .slices
            .values()
            .filter(|slice| !slice.is_free())
            .collect();

        MemoryUsage {
            number_allocs: used_slices.len() as u64,
            bytes_in_use: used_slices.iter().map(|slice| slice.storage.size()).sum(),
            bytes_padding: used_slices.iter().map(|slice| slice.padding).sum(),
            bytes_reserved: self.slices.values().map(|slice| slice.storage.size()).sum(),
        }
    }

    fn cleanup<Storage: crate::storage::ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        _alloc_nr: u64,
        explicit: bool,
    ) {
        if explicit {
            self.slices.retain(|_, slice| {
                if slice.is_free() {
                    storage.dealloc(slice.storage.id);
                    false
                } else {
                    true
                }
            });
        }
    }
}
