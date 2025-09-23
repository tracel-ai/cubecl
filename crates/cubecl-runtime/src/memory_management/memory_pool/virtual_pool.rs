use super::{MemoryPool, SliceBinding, SliceHandle, SliceId, MemoryPage, RingBuffer, calculate_padding};
use crate::memory_management::MemoryUsage;
use crate::server::IoError;
use crate::storage::{StorageHandle, StorageId, VirtualStorage};
use alloc::vec::Vec;
use hashbrown::HashMap;


struct DefragmentableMemoryPool {
    
    pages: HashMap<StorageId, MemoryPage>,
    ring: RingBuffer,
    storage_index: SearchIndex<StorageId>,
    /// Maps slice IDs to their corresponding slices
    slices: HashMap<SliceId, VirtualSlice>,
    /// Allocation_counter
    alloc_count: usize,
    /// Maximum allocation size supported
    max_alloc_size: u64,
    /// Memory alignment requirement.
    /// In practice, the virtual storage will align allocations to [`physical_block_size`]
    alignment: u64,
}

impl VirtualPool {
    fn new(max_alloc_size: u64, alignment: u64) -> Self {
        Self {

            slices: HashMap::new(),
            alloc_count: 0,
            max_alloc_size,
            alignment,
        }
    }

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
                    return Ok((Some(slice_id), false));
                }

                // Case 2. Slice is bigger, need to split
                if slice_size > required_size {
                    return Ok((Some(slice_id), true));
                }
            }
        }

        Ok((None, false))
    }
}

impl MemoryPool for VirtualPool {
    /// Retrieve the maximum allocation size of the pool
    fn max_alloc_size(&self) -> u64 {
        self.max_alloc_size
    }

    /// Get a new slice.
    fn get(&self, binding: &SliceBinding) -> Option<&StorageHandle> {
        self.slices.get(binding.id()).map(|slice| &slice.storage)
    }


    fn try_reserve(&mut self, size: u64) -> Option<SliceHandle> {None}

    /// Attempt to get a free slice.
   /* fn try_reserve(&mut self, size: u64) -> Option<SliceHandle> {
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
    }*/

    fn alloc<Storage: crate::storage::VirtualStorage>(
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
        self.alloc_count += 1;
        let storage_handle = storage.alloc(size)?; // Should allocate and map if using virtual memory

        if self.alloc_count > 10 {

            // Comprobar si storage es tambien vIRTUALSTORAGE
            self.defragment(storage);
        }

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

    fn cleanup<Storage: crate::storage::VirtualStorage>(
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

impl VirtualPool {


    fn defragment<Storage: VirtualStorage>(&mut self,  storage: &mut Storage) -> Result<(), IoError>
    {
        let all_free = self.get_free_slices();

        if let Some(free_slice_ids) = all_free {
            for slice_id in free_slice_ids {
                if let Some(mut slice) = self.slices.remove(&slice_id) {
                    storage.dealloc(slice.storage_id());
                    slice.set_mapped(false);
                }
            }
        }

        let final_handle = storage
            .defragment()
            .ok_or(IoError::InvalidHandle)?;

        let slice = self.create_slice(final_handle);
        self.slices.insert(slice.id(), slice);

        Ok(())
    }
}
