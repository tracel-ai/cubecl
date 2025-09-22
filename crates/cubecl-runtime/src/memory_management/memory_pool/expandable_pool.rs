use super::{MemoryPool,  SliceBinding, SliceHandle, SliceId, calculate_padding};
use crate::memory_management::MemoryUsage;
use crate::server::IoError;
use crate::storage::{StorageHandle, StorageId, VirtualStorage};
use alloc::vec::Vec;
use hashbrown::HashMap;
use std::collections::HashSet;


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
    fn new(
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

/// This memory pool manages multiple [`VirtualStorages`] of different block sizes.
/// The main purpose of having multiple backing storages is to reduce to potential extra padding
/// which can be associated with using virtual memory.
/// Each VirtualStorage allocates physical memory blocks of size [`physical_block_size`]
///
/// At runtime, the memory pool selects the most optimal storage to minimize this padding.
///
/// The rule is:
///    1. Prioritize the virtual storage with lower padding.
///    2. If there is a tie, prioritize the storage with the highest physical_block_size.ç
///
/// The main difference between this pool and the [`SlicedPool`] is that
/// memory defragmentation happens at the [`VirtualStorage`] level.
/// This allows for potentially more effective and finer grained merging and splitting of memory blocks.
///
/// # Memory allocation workflow:
///
/// At allocation time, the storage that minimizes the ratio padding / physical - block - size is chosen.
/// Inside the storage, the pool searches for all free slices.
/// If it found an exact match, returns the free slice of the exact size as the requested size.
/// If no exact match found, retrieves an slice that has a larger size than the requested size, calling the [`split_range`] method of the virtual storage to split the memory range at the required offset. Note that slices must be unmapped before this happens.
struct VirtualPool<V: VirtualStorage> {
    /// Maps block size to its corresponding VirtualStorage instance
    virtual_storages: HashMap<u64, V>,
    /// Maps slice IDs to their corresponding slices
    slices: HashMap<SliceId, VirtualSlice>,
    /// Maintains state about the location of each slice.
    storage_to_slice: HashMap<u64, HashSet<SliceId>>,
    /// Maximum allocation size supported
    max_alloc_size: u64,
    /// Memory alignment requirement.
    /// This should be the allocation granularity of the target device.
    /// However, in practice, each virtual storage will align allocations to [`physical_block_size`]
    alignment: u64,
    /// To keep track of free slices and where they are.
    free_slices: HashMap<u64, Vec<SliceId>>,
}

impl<V: VirtualStorage> VirtualPool<V> {
    fn new(
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

    /// Get a new slice.
    fn get(&self, binding: &SliceBinding) -> Option<&StorageHandle> {
        self.slices
            .get(binding.id())
            .map(|slice| &slice.storage)
    }

    /// Attempt to get a free slice.
    fn try_reserve(&mut self, size: u64) -> Option<SliceHandle> {
        // Runtime check to validate the size parameter
        if size == 0 {
            panic!(
                "Cannot allocate zero-sized memory");
        }

        if size > self.max_alloc_size {
            return None;
        }

        // Select the best fit storage for this reservation
        let block_size = self.select_optimal_block_size(size).expect("Storage hashmap not initialized!");

        // Try to reuse a free slice.
        if let (Some(reused_slice_id), need_to_split) = self.find_slice_in_storage( block_size, size).ok()? {


            // If the result of the find_slice algorithm indicates that we need to split the slice, split it using the virtual storage and insert the resulting slice in the free list.
            if need_to_split {
                let storage = self.virtual_storages.get_mut(&block_size).expect("Storage not found!");
                let slice = self.slices.get_mut(&reused_slice_id).expect("Slice not found");

                if let Ok(storage_handle) = storage.split_range(&mut slice.storage, size){

                    // Create a new slice and push it to the free list.
                    let new_slice = self.create_slice(storage_handle);
                    self.free_slices.entry(block_size).or_default().push(new_slice.id());
                    self.slices.insert(new_slice.id(), new_slice);
                }
            }


            // Map the slice if not already mapped
            let slice = self.slices.get_mut(&reused_slice_id).expect("Slice not found");
            if !slice.is_mapped() {
                let storage = self.virtual_storages.get_mut(&block_size).expect("Storage not found!");
                if storage.map(&mut slice.storage).is_err() {
                    return None
                };
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

        self.allocate_new_slice(size)
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
        let padding = calculate_padding(storage_handle.size(), block_size);


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
            .or_default()
            .insert(slice.id());

        slice
    }

    fn find_slice_in_storage(
        &mut self,
        block_size: u64,
        required_size: u64,
    ) -> Result<(Option<SliceId>, bool), IoError> {
        // Get the free list of slices of this storage.
        let free_list = match self.free_slices.get_mut(&block_size) {
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

    /// Alocates a completely new slice.
    fn allocate_new_slice(
        &mut self,
        size: u64,
    ) -> Result<SliceHandle, IoError> {
        let block_size = self
            .select_optimal_block_size(size)
            .ok_or_else(|| IoError::Unknown("No suitable virtual storage found".to_string()))?;
        let storage = self
            .virtual_storages
            .get_mut(&block_size)
            .ok_or_else(|| IoError::Unknown("Virtual storage not found".to_string()))?;


        // Reserve and map a range of virtual addresses using the virtual storage.
        let mut storage_handle = storage.reserve(size, 0u64)?;
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
            .get(&block_size).cloned()
            .unwrap_or_default();

        // Try to merge with any adjacent slice
        for &other_id in free_slice_ids.iter() {
            if other_id != slice_id && self.slices.contains_key(&other_id) && self.are_slices_adjacent(slice_id, other_id)? {
                    return self.merge_slice(slice_id, other_id);
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
            if let Some(free_list) = self.free_slices.get_mut(&block_size) && !free_list.contains(&slice_id) {
                    free_list.push(slice_id);
                }
            }

            // Try to merge with adjacent slices
            self.try_merge_with_adjacent(slice_id)?;
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
                .get(&block_size).cloned()
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

                for second_id in free_slice_ids.iter().skip(i + 1) {

                    // Skip if slice was already merged
                    if !self.slices.contains_key(second_id) {
                        continue;
                    }

                    // Check if slices are adjacent
                    if self.are_slices_adjacent(first_id, *second_id)? {
                        // Attempt to merge
                        match self.merge_slice(first_id, *second_id) {
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
