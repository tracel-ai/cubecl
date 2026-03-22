use super::{ManagedMemoryHandle, MemoryPool, Slice, calculate_padding};
use crate::memory_management::{BytesFormat, MemoryLocation};
use crate::{memory_management::MemoryUsage, server::IoError};
use alloc::vec;
use alloc::vec::Vec;
use cubecl_common::backtrace::BackTrace;
use hashbrown::HashMap;

pub struct PersistentPool {
    slices: Vec<Slice>,
    sizes: HashMap<u64, Vec<usize>>,
    alignment: u64,
    max_alloc_size: u64,
    location_base: MemoryLocation,
}

impl core::fmt::Display for PersistentPool {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for (size, positions) in self.sizes.iter() {
            let mut num_free = 0;
            let mut num_full = 0;
            let total = positions.len();

            for pos in positions {
                let slice = &self.slices[*pos];
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
    pub fn new(max_alloc_size: u64, alignment: u64, pool_pos: u8) -> Self {
        Self {
            slices: Vec::new(),
            sizes: HashMap::new(),
            max_alloc_size,
            alignment,
            location_base: MemoryLocation::new(pool_pos, 0, 0),
        }
    }

    pub fn has_size(&mut self, size: u64) -> bool {
        let padding = calculate_padding(size, self.alignment);
        let size_reserve = size + padding;
        self.sizes.contains_key(&size_reserve)
    }
}

impl MemoryPool for PersistentPool {
    fn accept(&self, size: u64) -> bool {
        self.max_alloc_size >= size
    }

    fn find(&self, binding: &super::ManagedMemoryBinding) -> Result<&Slice, IoError> {
        let slice_index = binding.descriptor().slice();

        self.slices
            .get(slice_index)
            .ok_or_else(|| IoError::NotFound {
                backtrace: BackTrace::capture(),
                reason: alloc::format!("Memory slice {} doesn't exist", slice_index).into(),
            })
    }

    fn try_reserve(&mut self, size: u64) -> Option<ManagedMemoryHandle> {
        let padding = calculate_padding(size, self.alignment);
        let size_reserve = size + padding;

        if let Some(positions) = self.sizes.get_mut(&size_reserve) {
            for pos in positions {
                let slice = &self.slices[*pos];

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
    ) -> Result<ManagedMemoryHandle, IoError> {
        let padding = calculate_padding(size, self.alignment);
        let size_alloc = size + padding;

        let storage_handle = storage.alloc(size_alloc)?;
        let slice = Slice::new(storage_handle, padding);
        let slice_id = slice.descriptor();
        let slice_pos = self.slices.len();
        let mut location = self.location_base;
        location.slice = slice_pos as u32;
        slice_id.update_location(location);

        match self.sizes.get_mut(&size) {
            Some(vals) => {
                vals.push(slice_pos);
            }
            None => {
                self.sizes.insert(size, vec![slice_pos]);
            }
        }

        let handle = slice.handle.clone();
        self.slices.push(slice);

        Ok(handle)
    }

    fn get_memory_usage(&self) -> MemoryUsage {
        let used_slices: Vec<_> = self
            .slices
            .iter()
            .filter(|slice| !slice.is_free())
            .collect();

        MemoryUsage {
            number_allocs: used_slices.len() as u64,
            bytes_in_use: used_slices.iter().map(|slice| slice.storage.size()).sum(),
            bytes_padding: used_slices.iter().map(|slice| slice.padding).sum(),
            bytes_reserved: self.slices.iter().map(|slice| slice.storage.size()).sum(),
        }
    }

    fn cleanup<Storage: crate::storage::ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        _alloc_nr: u64,
        explicit: bool,
    ) {
        if explicit {
            // We have to recompute all locations, so it's just safer to rebuild everything.
            let mut slices = Vec::new();
            let mut sizes = HashMap::<u64, Vec<usize>>::new();

            for slice in self.slices.drain(..) {
                if slice.is_free() {
                    storage.dealloc(slice.storage.id);
                } else {
                    let slice_pos = slices.len();
                    let size = slice.storage.size();
                    slice.descriptor().update_slice(slice_pos as u32);
                    slices.push(slice);

                    match sizes.get_mut(&size) {
                        Some(vals) => {
                            vals.push(slice_pos);
                        }
                        None => {
                            sizes.insert(size, vec![slice_pos]);
                        }
                    }
                }
            }

            self.sizes = sizes;
            self.slices = slices;
            storage.flush();
        }
    }

    fn bind(
        &mut self,
        old: ManagedMemoryHandle,
        new: ManagedMemoryHandle,
        cursor: u64,
    ) -> Result<(), IoError> {
        let slice = &mut self.slices[old.descriptor().slice()];
        new.descriptor()
            .update_location(old.descriptor().location());
        slice.cursor = cursor;
        slice.handle = new;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::storage::BytesStorage;

    use super::*;

    #[test_log::test]
    fn persistent_pool() {
        let mut storage = BytesStorage::default();
        let mut pool = PersistentPool::new(1024 * 1024, 4, 0);

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
