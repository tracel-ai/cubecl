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
    storage_to_slice: HashMap<u64, Vec<SliceId>>,
    /// Maximum allocation size supported
    max_alloc_size: u64,
    /// Memory alignment requirement.
    /// This should be the allocation granularity of the target device.
    /// However, in practice, each virtual storage will align allocations to [`physical_block_size`]
    alignment: u64,
}

impl<V: VirtualStorage> VirtualPool<V> {
    fn new(max_alloc_size: u64, alignment: u64, virtual_storages: HashMap<u64, V>) -> Self {
        let storage_to_slice: HashMap<u64, Vec<SliceId>> = virtual_storages
            .keys()
            .map(|&block_size| (block_size, Vec::new()))
            .collect();

        Self {
            virtual_storages,
            slices: HashMap::new(),
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

        // Select the best fit storage for this reservation
        let block_size = self
            .select_optimal_block_size(size)
            .expect("Storage hashmap not initialized!");

        // Try to reuse a free slice.
        if let (Some(reused_slice_id), need_to_split) =
            self.find_slice_in_storage(block_size, size).ok()?
        {
            // If the result of the find_slice algorithm indicates that we need to split the slice, split it using the virtual storage and insert the resulting slice in the free list.
            if need_to_split {
                let storage = self
                    .virtual_storages
                    .get_mut(&block_size)
                    .expect("Storage not found!");
                let slice = self
                    .slices
                    .get_mut(&reused_slice_id)
                    .expect("Slice not found");

                if let Ok(storage_handle) = storage.split_range(&mut slice.storage, size) {
                    // Create a new slice and push it to the free list.
                    let new_slice = self.create_slice(storage_handle, block_size);
                    self.slices.insert(new_slice.id(), new_slice);
                }
            }

            // Map the slice if not already mapped
            let slice = self
                .slices
                .get_mut(&reused_slice_id)
                .expect("Slice not found");
            if !slice.is_mapped() {
                let storage = self
                    .virtual_storages
                    .get_mut(&block_size)
                    .expect("Storage not found!");
                if storage.map(&mut slice.storage).is_err() {
                    return None;
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
        if explicit {}
    }
}

impl<V: VirtualStorage> VirtualPool<V> {
    /// Creates a virtual slice of size `size` upon the given page with the given offset.
    fn create_slice(&mut self, storage_handle: StorageHandle, block_size: u64) -> VirtualSlice {
        // Find the best virtual storage for this size
        let padding = calculate_padding(storage_handle.size(), block_size);

        assert_eq!(
            storage_handle.offset() % self.alignment,
            0,
            "slice with offset {} needs to be a multiple of {}",
            storage_handle.offset(),
            self.alignment
        );

        let slice_handle = SliceHandle::new();
        let mut slice = VirtualSlice::new(storage_handle, slice_handle, padding, block_size);

        slice.set_mapped(false);

        self.storage_to_slice
            .entry(block_size)
            .or_default()
            .push(slice.id());


        slice
    }

    fn get_free_slices_for_storage(&self, block_size: u64) -> Option<Vec<SliceId>> {
        let free_list: Vec<SliceId> = self
            .storage_to_slice
            .get(&block_size)?
            .iter()
            .filter_map(|&slice_id| {
                self.slices
                    .get(&slice_id)
                    .and_then(|slice| slice.is_free().then_some(slice_id))
            })
            .collect();

        (!free_list.is_empty()).then_some(free_list)
    }

    fn find_slice_in_storage(
        &mut self,
        block_size: u64,
        required_size: u64,
    ) -> Result<(Option<SliceId>, bool), IoError> {
        // Get the free list of slices of this storage.
        let mut free_list = match self.get_free_slices_for_storage(block_size) {
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
    fn allocate_new_slice(&mut self, size: u64) -> Result<SliceHandle, IoError> {
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

        let mut slice = self.create_slice(storage_handle, block_size);
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
            .max_by_key(|&block_size| (-(size as i64 % *block_size as i64), block_size))
            .or_else(|| {
                self.virtual_storages
                    .keys()
                    .min_by_key(|&&block_size| size % block_size)
            })
            .copied()
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

    fn defragment(&mut self) -> Result<(), IoError> {


        // Process each virtual storage separately
        for &block_size in self
            .virtual_storages
            .keys()
            .cloned()
            .collect::<Vec<_>>()
            .iter()
        {
            self.defragment_storage(block_size)?;
        }

        Ok(())
    }

    fn defragment_storage(&mut self, block_size: u64) -> Result<(), IoError> {


        let all_free = self.get_free_slices_for_storage(block_size);


        if let Some(storage_slices) = self.storage_to_slice.get_mut(&block_size) {
            if let Some(ref free_slice_ids) = all_free {
                storage_slices.retain(|id| !free_slice_ids.contains(id));
            }
        }

        if let Some(free_slice_ids) = all_free {
            for slice_id in free_slice_ids {
                if let Some(mut slice) = self.slices.remove(&slice_id) {
                    let storage = self
            .virtual_storages
            .get_mut(&block_size)
            .expect("Virtual storage hashmap is not initialized");
                    storage.unmap(slice.storage_id());
                    slice.set_mapped(false);
                }
            }
        }
        let storage = self
            .virtual_storages
            .get_mut(&block_size)
            .expect("Virtual storage hashmap is not initialized");
        let final_handle = storage.defragment().ok_or(IoError::InvalidHandle)?;
        self.create_slice(final_handle, block_size);
        Ok(())
    }
}
