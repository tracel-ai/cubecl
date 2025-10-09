use super::{MemoryPool, Slice, SliceHandle, SliceId, calculate_padding};
use crate::memory_management::BytesFormat;
use crate::{memory_management::MemoryUsage, server::IoError};
use alloc::vec;
use alloc::vec::Vec;
use hashbrown::HashMap;

pub struct PersistentPool {
    slices: HashMap<SliceId, Slice>,
    sizes: HashMap<u64, Vec<SliceId>>,
    alignment: u64,
    max_alloc_size: u64,
}

impl core::fmt::Display for PersistentPool {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for (size, ids) in self.sizes.iter() {
            let mut num_free = 0;
            let mut num_full = 0;
            let total = ids.len();

            for id in ids {
                let slice = self.slices.get(id).unwrap();
                let is_free = slice.is_free();
                if is_free {
                    num_free += 1;
                } else {
                    num_full += 1;
                }
            }

            f.write_fmt(format_args!(
                "  - Slices {} =>  {num_free} free - {num_full} full - {total} total\n",
                BytesFormat::new(*size)
            ))?;
        }

        if !self.sizes.is_empty() {
            f.write_fmt(format_args!("\n{}\n", self.get_memory_usage()))?;
        }

        Ok(())
    }
}

impl PersistentPool {
    pub fn new(max_alloc_size: u64, alignment: u64) -> Self {
        Self {
            slices: HashMap::new(),
            sizes: HashMap::new(),
            max_alloc_size,
            alignment,
        }
    }

    pub fn has_size(&mut self, size: u64) -> bool {
        let padding = calculate_padding(size, self.alignment);
        let size_reserve = size + padding;
        self.sizes.contains_key(&size_reserve)
    }
}

impl MemoryPool for PersistentPool {
    fn max_alloc_size(&self) -> u64 {
        self.max_alloc_size
    }

    fn get(&self, binding: &super::SliceBinding) -> Option<&crate::storage::StorageHandle> {
        self.slices.get(binding.id()).map(|slice| &slice.storage)
    }

    fn try_reserve(&mut self, size: u64) -> Option<SliceHandle> {
        let padding = calculate_padding(size, self.alignment);
        let size_reserve = size + padding;

        if let Some(vals) = self.sizes.get_mut(&size_reserve) {
            for id in vals {
                let slice = self.slices.get(id).unwrap();

                if slice.is_free() {
                    return Some(slice.handle.clone());
                }
            }
        }

        None
    }

    fn alloc<Storage: crate::storage::ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
    ) -> Result<SliceHandle, IoError> {
        let padding = calculate_padding(size, self.alignment);
        let size_alloc = size + padding;

        let storage_handle = storage.alloc(size_alloc)?;
        let slice_handle = SliceHandle::new();
        let slice = Slice::new(storage_handle, slice_handle.clone(), padding);

        let slice_id = slice.id();

        match self.sizes.get_mut(&size) {
            Some(vals) => {
                vals.push(slice_id);
            }
            None => {
                self.sizes.insert(size, vec![slice_id]);
            }
        }

        self.slices.insert(slice_id, slice);

        Ok(slice_handle)
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
            let mut removed = Vec::new();
            self.slices.retain(|id, slice| {
                if slice.is_free() {
                    storage.dealloc(slice.storage.id);
                    removed.push((*id, slice.effective_size()));
                    false
                } else {
                    true
                }
            });

            for (id, size) in removed {
                let ids = self.sizes.get_mut(&size).expect("The size should match");
                ids.retain(|id_| *id_ != id);
            }

            storage.flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::storage::BytesStorage;

    use super::*;

    #[test]
    fn persistent_pool() {
        let mut storage = BytesStorage::default();
        let mut pool = PersistentPool::new(1024 * 1024, 4);

        let result = pool.try_reserve(1024);
        assert!(result.is_none(), "No alloc yet");

        let alloc1 = pool.alloc(&mut storage, 1024);
        let result = pool.try_reserve(1024);
        assert!(result.is_none(), "No free slice yet, handle1 is alive");

        core::mem::drop(alloc1);
        let result = pool.try_reserve(1024);
        assert!(result.is_some(), "Handle1 is free to be reused.");
        core::mem::drop(result);

        let result = pool.try_reserve(1025);
        assert!(result.is_none(), "Not the same size.");

        let alloc2 = pool.alloc(&mut storage, 1024);
        let usage = pool.get_memory_usage();
        assert_eq!(usage.bytes_in_use, 1024);
        assert_eq!(usage.bytes_reserved, 2048);

        let result = pool.try_reserve(1024);
        let usage = pool.get_memory_usage();
        assert!(result.is_some(), "Handle1 is free to be reused.");
        assert_eq!(usage.bytes_in_use, 2048);
        assert_eq!(usage.bytes_reserved, 2048);

        core::mem::drop(alloc2);
        core::mem::drop(result);

        let usage = pool.get_memory_usage();
        assert_eq!(usage.bytes_in_use, 0);
        assert_eq!(usage.bytes_reserved, 2048);
    }
}
