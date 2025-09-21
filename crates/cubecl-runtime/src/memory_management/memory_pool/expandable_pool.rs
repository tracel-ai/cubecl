use super::{MemoryPool, RingBuffer, Slice, SliceBinding, SliceHandle, SliceId, calculate_padding};
use crate::memory_management::MemoryUsage;
use crate::server::IoError;
use crate::storage::{StorageHandle, StorageId, VirtualStorage};
use alloc::vec::Vec;
use hashbrown::HashMap;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub(crate) struct VirtualSlice {
    pub storage: StorageHandle,
    pub handle: SliceHandle,
    pub padding: u64,
    pub block_size: u64,
    pub is_mapped: bool,
}

impl VirtualSlice {
    pub(crate) fn new(
        storage: StorageHandle,
        handle: SliceHandle,
        padding: u64,
        block_size: u64,
    ) -> Self {
        Self {
            storage,
            handle,
            padding,
            block_size,
            is_mapped: false,
        }
    }

    pub(crate) fn is_free(&self) -> bool {
        self.handle.is_free()
    }

    pub(crate) fn effective_size(&self) -> u64 {
        self.storage.size() + self.padding
    }

    pub(crate) fn id(&self) -> SliceId {
        *self.handle.id()
    }

    pub(crate) fn block_size(&self) -> u64 {
        self.block_size
    }

    pub(crate) fn is_mapped(&self) -> bool {
        self.is_mapped
    }

    pub(crate) fn set_mapped(&mut self, mapped: bool) {
        self.is_mapped = mapped;
    }

    pub(crate) fn storage_size(&self) -> u64 {
        self.storage.size()
    }

    pub(crate) fn storage_id(&self) -> StorageId {
        self.storage.id
    }
}

/// A memory pool that manages multiple VirtualStorage instances with different block sizes
/// to optimize memory allocation efficiency and reduce fragmentation.
pub(crate) struct VirtualPool<V: VirtualStorage> {
    /// Maps block size to its corresponding VirtualStorage instance
    virtual_storages: HashMap<u64, V>,
    /// Maps slice IDs to their corresponding slices
    slices: HashMap<SliceId, VirtualSlice>,
    /// To query the slices on each storage.
    storage_to_slice: HashMap<u64, HashSet<SliceId>>,

    /// Maximum allocation size supported
    max_alloc_size: u64,
    /// Memory alignment requirement
    alignment: u64,
    //
    free_slices: HashMap<u64, Vec<SliceId>>,
}

impl<V: VirtualStorage> VirtualPool<V> {
    pub(crate) fn new(
        max_alloc_size: u64,
        alignment: u64,
        virtual_storages: HashMap<u64, V>,
    ) -> Self {
        let storage_to_slice: HashMap<u64, HashSet<SliceId>> = virtual_storages
            .keys()
            .map(|&block_size| (block_size, HashSet::new()))
            .collect();
        let free_slices: HashMap<u64, Vec<SliceId>> = virtual_storages
            .keys()
            .map(|&block_size| (block_size, Vec::new()))
            .collect();
        Self {
            virtual_storages,
            slices: HashMap::new(),
            free_slices,
            storage_to_slice,
            max_alloc_size,
            alignment,
        }
    }
}

impl<V: VirtualStorage> MemoryPool for VirtualPool<V> {
    /// Retrieve the maximum allocation size of the pool
    fn max_alloc_size(&self) -> u64 {
        self.max_alloc_size
    }

    fn get(&self, binding: &SliceBinding) -> Option<&StorageHandle> {
        self.slices
            .get(binding.id())
            .map(|slice| &slice.storage)
    }

    /// Attempt to get a free slice.
    fn try_reserve(&mut self, size: u64) -> Option<SliceHandle> {
        if size == 0 {
            panic!(
                "Cannot allocate zero-sized memory");
        }

        if size > self.max_alloc_size {
            return None;
        }

        let padding = calculate_padding(size, self.alignment);
        let effective_size = size + padding;
        let block_size = self.select_optimal_block_size(effective_size).expect("Storage hashmap not initialized!");

        if let (Some(reused_slice_id), need_to_split) = self.find_slice_in_storage( block_size, effective_size).ok()? {


            if need_to_split {
                let storage = self.virtual_storages.get_mut(&block_size).expect("Storage not found!");
                let slice = self.slices.get_mut(&reused_slice_id).expect("Slice not found");

                if let Ok(storage_handle) = storage.split_range(&mut slice.storage, effective_size as usize){
                    let new_slice = self.create_slice(storage_handle);
                    self.free_slices.entry(block_size).or_insert_with(Vec::new).push(new_slice.id());
                    self.slices.insert(new_slice.id(), new_slice);
                }
            }

            let slice = self.slices.get_mut(&reused_slice_id).expect("Slice not found");

            if !slice.is_mapped() {
                let storage = self.virtual_storages.get_mut(&block_size).expect("Storage not found!");
                storage.map(&mut slice.storage);
            };

            return Some(slice.handle.clone());
        }

        None
    }

    fn alloc<Storage: crate::storage::ComputeStorage>(
        &mut self,
        _storage: &mut Storage,
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

        let padding = calculate_padding(size, self.alignment);
        let effective_size = size + padding;
        self.allocate_new_slice(effective_size, padding)
    }

    fn get_memory_usage(&self) -> MemoryUsage {
        //// TODO
        unimplemented!()
    }

    fn cleanup<Storage: crate::storage::ComputeStorage>(
        &mut self,
        _storage: &mut Storage,
        _alloc_nr: u64,
        explicit: bool,
    ) {
        if explicit {

        }
    }
}

impl<V: VirtualStorage> VirtualPool<V> {


    /// Creates a virtual slice of size `size` upon the given page with the given offset.
    fn create_slice(&mut self, storage_handle: StorageHandle) -> VirtualSlice {
        // Find the best virtual storage for this size
        let block_size = self.select_optimal_block_size(storage_handle.size()).expect("Storage hashmap not initialized!");
        let storage = self.virtual_storages.get_mut(&block_size).expect("Storage not found!");

        let padding = calculate_padding(storage_handle.size(), block_size);
        let effective_size = storage_handle.size() + padding;

        assert_eq!(
            storage_handle.offset() % self.alignment,
            0,
            "slice with offset {} needs to be a multiple of {}",
            storage_handle.offset(), self.alignment
        );

        let slice_handle = SliceHandle::new();
        let mut slice = VirtualSlice::new(
           storage_handle, slice_handle, padding,
            block_size,
        );


        slice.set_mapped(false);
        // We do not push to the freelist yet. That must be done on purpose.

        self.storage_to_slice
            .entry(block_size)
            .or_insert_with(HashSet::new)
            .insert(slice.id());

        slice
    }

    fn find_slice_in_storage(
        &mut self,
        block_size: u64,
        required_size: u64,
    ) -> Result<(Option<SliceId>, bool), IoError> {
        let free_list = match self.free_slices.get_mut(&block_size) {
            Some(list) => list,
            None => return Ok((None, false)),
        };

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

    /// Alocates a completely new slice.
    fn allocate_new_slice(
        &mut self,
        effective_size: u64,
        padding: u64,
    ) -> Result<SliceHandle, IoError> {
        let block_size = self
            .select_optimal_block_size(effective_size)
            .ok_or_else(|| IoError::Unknown("No suitable virtual storage found".to_string()))?;
        let storage = self
            .virtual_storages
            .get_mut(&block_size)
            .ok_or_else(|| IoError::Unknown("Virtual storage not found".to_string()))?;

        let mut storage_handle = storage.reserve(effective_size as usize, 0u64)?;
        storage.map(&mut storage_handle)?;

        let mut slice = self.create_slice(storage_handle);
        slice.set_mapped(true);
        let handle = slice.handle.clone();
        self.slices.insert(slice.id(), slice);

        Ok(handle)
    }

    /// Selects the optimal block size for a given allocation size
    // Only returns None if the virtual storage hashmap has not yet been initialized
    fn select_optimal_block_size(&self, size: u64) -> Option<u64> {
        // Find storage with block size that minimizes waste
        // Prefer larger block sizes for large allocations to reduce API calls
         self.virtual_storages
        .keys()
        .filter(|&&block_size| block_size <= size)
        .max_by_key(|&block_size| {
            (-(size as i64 % *block_size as i64), block_size)
        })
        .or_else(|| {
            self.virtual_storages
                .keys()
                .min_by_key(|&&block_size| size % block_size)
        })
        .copied()
    }

    /// Expands an existing free slice to a new size, allocating contiguous virtual memory.
    pub fn expand_slice(&mut self, slice_id: SliceId, new_size: u64) -> Result<(), IoError> {
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

        let block_size = slice.block_size();
        let padding = calculate_padding(new_size, block_size);
        let effective_new_size = new_size + padding;
        let additional_size = effective_new_size - current_size;

        let storage = self
            .virtual_storages
            .get_mut(&block_size)
            .ok_or_else(|| IoError::Unknown("Virtual storage not found".to_string()))?;

        storage.expand(&mut slice.storage, additional_size)
    }


       /// Attempts to merge a slice with any adjacent free slice.
    /// Useful when freeing a slice to immediately reduce fragmentation.
    pub fn try_merge_with_adjacent(&mut self, slice_id: SliceId) -> Result<SliceId, IoError> {
        let slice = self.slices.get(&slice_id)
            .ok_or_else(|| IoError::Unknown("Slice not found".to_string()))?;

        if !slice.is_free() {
            return Err(IoError::InvalidHandle);
        }

        let block_size = slice.block_size();
        let free_slice_ids: Vec<SliceId> = self.free_slices
            .get(&block_size)
            .map(|list| list.clone())
            .unwrap_or_default();

        // Try to merge with any adjacent slice
        for &other_id in free_slice_ids.iter() {
            if other_id != slice_id && self.slices.contains_key(&other_id) {
                if self.are_slices_adjacent(slice_id, other_id)? {
                    return self.merge_slice(slice_id, other_id);
                }
            }
        }

        // No adjacent slice found
        Ok(slice_id)
    }



        pub fn free_slice(&mut self, slice_id: SliceId) -> Result<(), IoError> {
        // Mark slice as free (implementation depends on your SliceHandle design)
        if let Some(slice) = self.slices.get_mut(&slice_id) {
            if slice.is_mapped() {
                let block_size = slice.block_size();
                let storage = self.virtual_storages.get_mut(&block_size)
                    .ok_or_else(|| IoError::Unknown("Virtual storage not found".to_string()))?;

                storage.unmap(slice.storage_id());
                slice.set_mapped(false);
            }

            // Add to free list if not already there
            let block_size = slice.block_size();
            if let Some(free_list) = self.free_slices.get_mut(&block_size) {
                if !free_list.contains(&slice_id) {
                    free_list.push(slice_id);
                }
            }

            // Try to merge with adjacent slices
            self.try_merge_with_adjacent(slice_id)?;
        }

        Ok(())
    }
 pub fn merge_slice(&mut self, first_id: SliceId, second_id: SliceId) -> Result<SliceId, IoError> {
        // Basic validations
        let first_slice = self.slices.get(&first_id)
            .ok_or_else(|| IoError::Unknown("First slice not found".to_string()))?;
        let second_slice = self.slices.get(&second_id)
            .ok_or_else(|| IoError::Unknown("Second slice not found".to_string()))?;

        if !first_slice.is_free() || !second_slice.is_free() {
            return Err(IoError::InvalidHandle);
        }

        if first_slice.is_mapped() || second_slice.is_mapped() {
            return Err(IoError::Unknown("Cannot merge mapped slices".to_string()));
        }

        let block_size = first_slice.block_size();
        if block_size != second_slice.block_size() {
            return Err(IoError::Unknown("Cannot merge slices from different virtual storages".to_string()));
        }

        // Remove both slices from pool data structures
        let first_slice = self.slices.remove(&first_id).unwrap();
        let second_slice = self.slices.remove(&second_id).unwrap();

        // Update storage_to_slice mapping
        if let Some(slice_set) = self.storage_to_slice.get_mut(&block_size) {
            slice_set.remove(&first_id);
            slice_set.remove(&second_id);
        }

        // Remove from free slices list
        if let Some(free_list) = self.free_slices.get_mut(&block_size) {
            free_list.retain(|&id| id != first_id && id != second_id);
        }

        // Delegate to VirtualStorage for the actual merge
        let storage = self.virtual_storages.get_mut(&block_size)
            .ok_or_else(|| IoError::Unknown("Virtual storage not found".to_string()))?;

        let merged_storage_handle = storage.merge(first_slice.storage, second_slice.storage)?;

        // Create new merged slice
        let merged_slice = self.create_slice(merged_storage_handle);
        let merged_id = merged_slice.id();
        self.slices.insert(merged_id, merged_slice);

        // Add to free list
        self.free_slices
            .entry(block_size)
            .or_insert_with(Vec::new)
            .push(merged_id);

        Ok(merged_id)
    }


     fn are_slices_adjacent(&self, first_id: SliceId, second_id: SliceId) -> Result<bool, IoError> {
        let first_slice = self.slices.get(&first_id)
            .ok_or_else(|| IoError::Unknown("First slice not found".to_string()))?;
        let second_slice = self.slices.get(&second_id)
            .ok_or_else(|| IoError::Unknown("Second slice not found".to_string()))?;

        // Must be from the same virtual storage
        if first_slice.block_size() != second_slice.block_size() {
            return Ok(false);
        }

        // Both must be free and unmapped
        if !first_slice.is_free() || !second_slice.is_free() {
            return Ok(false);
        }

        if first_slice.is_mapped() || second_slice.is_mapped() {
            return Ok(false);
        }

        let block_size = first_slice.block_size();
        let storage = self.virtual_storages.get(&block_size)
            .ok_or_else(|| IoError::Unknown("Virtual storage not found".to_string()))?;

        Ok(storage.are_adjacent(&first_slice.storage, &second_slice.storage))
    }



  pub fn defragment(&mut self) -> Result<u32, IoError> {
        let mut total_merges = 0;

        // Process each virtual storage separately
        for &block_size in self.virtual_storages.keys().cloned().collect::<Vec<_>>().iter() {
            total_merges += self.defragment_storage(block_size)?;
        }

        Ok(total_merges)
    }

      fn defragment_storage(&mut self, block_size: u64) -> Result<u32, IoError> {
        let mut merges_count = 0;
        let mut made_progress = true;

        // Keep trying to merge until no more merges are possible
        while made_progress {
            made_progress = false;

            // Get all free slices for this storage
            let free_slice_ids: Vec<SliceId> = self.free_slices
                .get(&block_size)
                .map(|list| list.clone())
                .unwrap_or_default();

            // Try to find adjacent pairs to merge
            for i in 0..free_slice_ids.len() {
                if made_progress {
                    break; // Restart after each successful merge
                }

                let first_id = free_slice_ids[i];

                // Skip if slice was already merged in this iteration
                if !self.slices.contains_key(&first_id) {
                    continue;
                }

                for j in (i + 1)..free_slice_ids.len() {
                    let second_id = free_slice_ids[j];

                    // Skip if slice was already merged
                    if !self.slices.contains_key(&second_id) {
                        continue;
                    }

                    // Check if slices are adjacent
                    if self.are_slices_adjacent(first_id, second_id)? {
                        // Attempt to merge
                        match self.merge_slice(first_id, second_id) {
                            Ok(_) => {
                                merges_count += 1;
                                made_progress = true;
                                break; // Restart the outer loop
                            },
                            Err(_) => {
                                // Merge failed, continue looking
                                continue;
                            }
                        }
                    }
                }
            }
        }

        Ok(merges_count)
    }



}
