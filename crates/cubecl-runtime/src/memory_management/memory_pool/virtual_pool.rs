use super::{MemoryPool, SliceBinding, SliceHandle, SliceId, calculate_padding};
use crate::memory_management::MemoryUsage;
use crate::server::IoError;
use crate::storage::{StorageHandle, StorageId, VirtualStorage};
use alloc::vec::Vec;
use hashbrown::HashMap;

/// An slice represents a virtual memory address range.
/// This slice can be either mapped or unmapped to physical memory.
/// At allocation time, [`VirtualSlices`] are automatically mapped, to guarantee it is available for use.
/// When the [`SliceHandle`] becomes free, [`VirtualSlices`] can be unmapped, allowing for potential memory defragmentation.
#[derive(Debug, Clone)]
struct VirtualSlice {
    pub storage: StorageHandle,
    pub handle: SliceHandle,
    pub padding: u64,
    pub block_size: u64,
    pub is_mapped: bool,
}

impl VirtualSlice {
    fn new(storage: StorageHandle, handle: SliceHandle, padding: u64, block_size: u64) -> Self {
        Self {
            storage,
            handle,
            padding,
            block_size,
            is_mapped: false,
        }
    }

    fn is_free(&self) -> bool {
        self.handle.is_free()
    }

    fn effective_size(&self) -> u64 {
        self.storage.size() + self.padding
    }

    fn id(&self) -> SliceId {
        *self.handle.id()
    }

    fn block_size(&self) -> u64 {
        self.block_size
    }

    fn is_mapped(&self) -> bool {
        self.is_mapped
    }

    fn set_mapped(&mut self, mapped: bool) {
        self.is_mapped = mapped;
    }

    fn storage_size(&self) -> u64 {
        self.storage.size()
    }

    fn storage_id(&self) -> StorageId {
        self.storage.id
    }
}

/// This memory pool operates on a [`VirtualStorages`] of a specific block size
/// A VirtualStorage allocates physical memory blocks of size [`physical_block_size`]
///
/// At runtime, the memory management shoukd select the virtual pool with most optimal storage to minimize this padding.
///
/// The rule is:
///    1. Prioritize the virtual storage with lower padding.
///    2. If there is a tie, prioritize the storage with the highest physical_block_size.
///
/// Look at the memory management module for details on this.
///
/// The main difference between this pool and the [`SlicedPool`] is that
/// memory defragmentation happens at the [`VirtualStorage`] level.
/// This allows for potentially more effective and finer grained merging and splitting of memory blocks.
///
/// # Memory allocation workflow:
///
/// Inside the storage, the pool searches for all free slices.
/// If it found an exact match, returns the free slice of the exact size as the requested size.
/// If no exact match found, retrieves an slice that has a larger size than the requested size, calling the [`split_range`] method of the virtual storage to split the memory range at the required offset. Note that slices must be unmapped before this happens.
struct VirtualPool<V: VirtualStorage> {
    /// Maps block size to its corresponding VirtualStorage instance
    virtual_storage: V,
    /// Maps slice IDs to their corresponding slices
    slices: HashMap<SliceId, VirtualSlice>,
    /// Maximum allocation size supported
    max_alloc_size: u64,
    /// Memory alignment requirement.
    /// In practice, the virtual storage will align allocations to [`physical_block_size`]
    alignment: u64,
}

impl<V: VirtualStorage> VirtualPool<V> {
    fn new(max_alloc_size: u64, alignment: u64, virtual_storage: V) -> Self {
        Self {
            virtual_storage,
            slices: HashMap::new(),
            max_alloc_size,
            alignment,
        }
    }
}

impl<VStorage: VirtualStorage> MemoryPool for VirtualPool<VStorage> {
    /// Retrieve the maximum allocation size of the pool
    fn max_alloc_size(&self) -> u64 {
        self.max_alloc_size
    }

    /// Get a new slice.
    fn get(&self, binding: &SliceBinding) -> Option<&StorageHandle> {
        self.slices.get(binding.id()).map(|slice| &slice.storage)
    }

    /// Attempt to get a free slice.
    fn try_reserve(&mut self, size: u64) -> Option<SliceHandle> {
        // Runtime check to validate the size parameter
        if size == 0 {
            panic!("Cannot allocate zero-sized memory");
        }

        if size > self.max_alloc_size {
            return None;
        }

        // Try to reuse a free slice.
        if let (Some(reused_slice_id), need_to_split) = self.find_slice(size).ok()? {
            // If the result of the find_slice algorithm indicates that we need to split the slice, split it using the virtual storage and insert the resulting slice in the free list.
            if need_to_split {
                let slice = self
                    .slices
                    .get_mut(&reused_slice_id)
                    .expect("Slice not found");

                if let Ok(storage_handle) =
                    self.virtual_storage.split_range(&mut slice.storage, size)
                {
                    // Create a new slice and push it to the free list.
                    let new_slice = self.create_slice(storage_handle);
                    self.slices.insert(new_slice.id(), new_slice);
                }
            }

            // Map the slice if not already mapped
            let slice = self
                .slices
                .get_mut(&reused_slice_id)
                .expect("Slice not found");
            if !slice.is_mapped() {
                if self.virtual_storage.map(&mut slice.storage).is_err() {
                    return None;
                };
            };

            return Some(slice.handle.clone());
        }

        None
    }

    fn alloc<Storage: crate::storage::ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
    ) -> Result<SliceHandle, IoError> {
        if size == 0 {
            return Err(IoError::Unknown(
                "Cannot allocate zero-sized memory".to_string(),
            ));
        }

        if size > self.max_alloc_size {
            return Err(IoError::BufferTooBig(size as usize));
        }

        // Reserve and map a range of virtual addresses using the virtual storage.
        let storage_handle = storage.alloc(size)?; // Should allocate and map if using virtual memory
        let mut slice = self.create_slice(storage_handle);
        slice.set_mapped(true);
        let handle = slice.handle.clone();
        self.slices.insert(slice.id(), slice);

        Ok(handle)
    }

    fn get_memory_usage(&self) -> MemoryUsage {
        //// TODO
        unimplemented!()
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
                    storage.dealloc(slice.storage.id); // unmaps
                    false
                } else {
                    true
                }
            });

            storage.flush(); // cleans up
        }
    }
}

impl<V: VirtualStorage> VirtualPool<V> {
    // Creates a virtual slice of size `size` upon the given page with the given offset.
    fn create_slice(&mut self, storage_handle: StorageHandle) -> VirtualSlice {
        // Find the best virtual storage for this size
        let padding = calculate_padding(storage_handle.size(), self.alignment);

        assert_eq!(
            storage_handle.offset() % self.alignment,
            0,
            "slice with offset {} needs to be a multiple of {}",
            storage_handle.offset(),
            self.alignment
        );

        let slice_handle = SliceHandle::new();
        let mut slice = VirtualSlice::new(storage_handle, slice_handle, padding, self.alignment);

        slice.set_mapped(false);

        slice
    }

    fn get_free_slices(&self) -> Option<Vec<SliceId>> {

        let free_list: Vec<SliceId> = self
            .slices
            .values()
            .filter(|slice| slice.is_free())
            .map(|slice| slice.id())
            .into_iter()
            .collect();

        (!free_list.is_empty()).then_some(free_list)
    }

    fn find_slice(&mut self, required_size: u64) -> Result<(Option<SliceId>, bool), IoError> {
        // Get the free list of slices of this storage.
        let mut free_list = match self.get_free_slices() {
            Some(list) => list,
            None => return Ok((None, false)),
        };

        // Iterate over the free list and check if we can
        for (index, &slice_id) in free_list.iter().enumerate() {
            if let Some(slice) = self.slices.get_mut(&slice_id) {
                if !slice.is_free() {
                    continue;
                }

                let slice_size = slice.storage_size();

                // Case 1: Match, no need to split.
                if slice_size == required_size {
                    free_list.remove(index);
                    return Ok((Some(slice_id), false));
                }

                // Case 2. Slice is bigger, need to split
                if slice_size > required_size {
                    free_list.remove(index);
                    return Ok((Some(slice_id), true));
                }
            }
        }

        Ok((None, false))
    }

    /// Expands an existing free slice to a new size, allocating contiguous virtual memory.
    fn _expand_slice(&mut self, slice_id: SliceId, new_size: u64) -> Result<(), IoError> {
        let slice = self
            .slices
            .get_mut(&slice_id)
            .ok_or_else(|| IoError::Unknown("Slice not found".to_string()))?;

        if !slice.is_free() {
            return Err(IoError::InvalidHandle);
        }

        let current_size = slice.storage_size();

        if new_size <= current_size {
            return Ok(());
        }

        let padding = calculate_padding(new_size, self.alignment);
        let effective_new_size = new_size + padding;
        let additional_size = effective_new_size - current_size;

        self.virtual_storage
            .expand(&mut slice.storage, additional_size)
    }

    fn defragment(&mut self) -> Result<(), IoError> {
        let all_free = self.get_free_slices();

        if let Some(free_slice_ids) = all_free {
            for slice_id in free_slice_ids {
                if let Some(mut slice) = self.slices.remove(&slice_id) {
                    self.virtual_storage.unmap(slice.storage_id());
                    slice.set_mapped(false);
                }
            }
        }

        let final_handle = self
            .virtual_storage
            .defragment()
            .ok_or(IoError::InvalidHandle)?;
        let slice = self.create_slice(final_handle);
        self.slices.insert(slice.id(), slice);

        Ok(())
    }
}
