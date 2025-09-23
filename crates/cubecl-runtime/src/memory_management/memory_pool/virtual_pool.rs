use super::{MemoryPool, SliceBinding, SliceHandle, SliceId, MemoryPage, RingBuffer, calculate_padding};
use super::index::SearchIndex;
use crate::memory_management::MemoryUsage;
use crate::server::IoError;
use crate::storage::{StorageHandle, StorageId, StorageUtilization, VirtualStorage};
use alloc::vec::Vec;
use hashbrown::HashMap;

/// Represents a virtual address space that acts as a "page" containing multiple slices
#[derive(Debug)]
pub(crate) struct VirtualAddressSpace {
    /// The virtual storage handle for this address space
    pub(crate) storage_handle: StorageHandle,
    /// Maps offset within this address space to slice IDs
    pub(crate) slices: HashMap<u64, SliceId>,
    /// Total size of this virtual address space
    pub(crate) total_size: u64,
}

impl VirtualAddressSpace {
    pub(crate) fn new(storage_handle: StorageHandle, total_size: u64) -> Self {
        Self {
            storage_handle,
            slices: HashMap::new(),
            total_size,
        }
    }

    /// Find slice at a specific offset within this address space
    pub(crate) fn find_slice(&self, offset: u64) -> Option<SliceId> {
        self.slices.get(&offset).copied()
    }

    /// Insert a slice at a specific offset within this address space
    pub(crate) fn insert_slice(&mut self, offset: u64, slice_id: SliceId) {
        self.slices.insert(offset, slice_id);
    }

    /// Remove a slice from this address space
    pub(crate) fn remove_slice(&mut self, offset: u64) -> Option<SliceId> {
        self.slices.remove(&offset)
    }

    /// Check if this address space is empty (no slices)
    pub(crate) fn is_empty(&self) -> bool {
        self.slices.is_empty()
    }
}

/// A slice that maps to a specific physical handle within a virtual address space
#[derive(Debug)]
pub(crate) struct VirtualSlice {
    /// Handle to the virtual address space this slice belongs to
    pub(crate) virtual_storage_id: StorageId,
    /// Offset within the virtual address space
    pub(crate) virtual_offset: u64,
    /// Size of the slice (without padding)
    pub(crate) size: u64,
    /// Padding for alignment
    pub(crate) padding: u64,
    /// The slice handle
    pub(crate) handle: SliceHandle,
    /// Whether this slice is currently mapped to physical memory
    pub(crate) mapped: bool,
    /// Physical handle if mapped (None if unmapped)
    pub(crate) physical_handle: Option<StorageHandle>,
}

impl VirtualSlice {
    pub(crate) fn new(
        virtual_storage_id: StorageId,
        virtual_offset: u64,
        size: u64,
        padding: u64,
        handle: SliceHandle,
    ) -> Self {
        Self {
            virtual_storage_id,
            virtual_offset,
            size,
            padding,
            handle,
            mapped: false,
            physical_handle: None,
        }
    }

    pub(crate) fn id(&self) -> SliceId {
        self.handle.id()
    }

    pub(crate) fn is_free(&self) -> bool {
        self.handle.is_free()
    }

    pub(crate) fn is_mapped(&self) -> bool {
        self.mapped
    }

    pub(crate) fn effective_size(&self) -> u64 {
        self.size + self.padding
    }

    pub(crate) fn storage_size(&self) -> u64 {
        self.size
    }

    /// Map this slice to physical memory
    pub(crate) fn map<Storage: VirtualStorage>(
        &mut self,
        storage: &mut Storage,
        virtual_spaces: &HashMap<StorageId, VirtualAddressSpace>,
    ) -> Result<(), IoError> {
        if self.mapped {
            return Ok(()); // Already mapped
        }

        // Map the region within the virtual address space to physical memory
        let physical_handle = storage.map(
            self.virtual_storage_id,
            self.virtual_offset,
            self.effective_size(),
        )?;

        self.physical_handle = Some(physical_handle);
        self.mapped = true;
        Ok(())
    }

    /// Unmap this slice from physical memory
    pub(crate) fn unmap<Storage: VirtualStorage>(&mut self, storage: &mut Storage) {
        if !self.mapped {
            return; // Already unmapped
        }

        storage.unmap(self.virtual_storage_id, self.virtual_offset, self.effective_size());
        self.physical_handle = None;
        self.mapped = false;
    }

    /// Split this slice at the given offset, creating a new slice for the remainder
    pub(crate) fn split<Storage: VirtualStorage>(
        &mut self,
        split_size: u64,
        alignment: u64,
        storage: &mut Storage,
    ) -> Result<Option<Self>, IoError> {
        if split_size >= self.effective_size() || split_size < alignment {
            return Ok(None);
        }

        let remaining_size = self.effective_size() - split_size;
        if remaining_size < alignment {
            // Can't create a valid slice from the remainder, add to padding instead
            self.padding = self.effective_size() - split_size;
            return Ok(None);
        }

        // Create new slice for the remainder
        let new_virtual_offset = self.virtual_offset + split_size;
        let new_size = remaining_size - calculate_padding(remaining_size, alignment);
        let new_padding = calculate_padding(remaining_size, alignment);
        let new_handle = SliceHandle::new();

        let mut new_slice = VirtualSlice::new(
            self.virtual_storage_id,
            new_virtual_offset,
            new_size,
            new_padding,
            new_handle,
        );

        // Handle physical memory mapping if original slice was mapped
        if self.mapped {
            // Unmap the portion that will belong to the new slice
            storage.unmap(
                self.virtual_storage_id,
                new_virtual_offset,
                remaining_size
            );

            // Map the new slice's portion to get its own physical handle
            let new_physical_handle = storage.map(
                self.virtual_storage_id,
                new_virtual_offset,
                remaining_size,
            )?;

            new_slice.physical_handle = Some(new_physical_handle);
            new_slice.mapped = true;
        }

        // Update this slice (reduce its size, remove padding)
        self.size = split_size;
        self.padding = 0;
        Ok(Some(new_slice))
    }

    /// Get the storage handle for external access
    pub(crate) fn get_storage_handle(&self) -> Option<StorageHandle> {
        self.physical_handle.clone()
    }
}

/// A defragmentable memory pool using virtual address spaces as pages
pub(crate) struct DefragmentableMemoryPool {
    /// Maps storage IDs to virtual address spaces (acting as pages)
    virtual_spaces: HashMap<StorageId, VirtualAddressSpace>,
    /// Maps slice IDs to their virtual slices
    slices: HashMap<SliceId, VirtualSlice>,
    /// Ring buffer for managing virtual address space allocation order
    ring: RingBuffer,
    /// Index for searching through virtual spaces
    storage_index: SearchIndex<StorageId>,
    /// Counter for tracking allocations
    alloc_count: usize,
    /// Threshold for triggering defragmentation
    defrag_threshold: usize,
    /// Maximum allocation size supported
    max_alloc_size: u64,
    /// Memory alignment requirement
    alignment: u64,
}

impl DefragmentableMemoryPool {
    pub(crate) fn new(max_alloc_size: u64, alignment: u64) -> Self {

        Self {
            virtual_spaces: HashMap::new(),
            slices: HashMap::new(),
            ring: RingBuffer::new(alignment),
            storage_index: SearchIndex::new(),
            alloc_count: 0,
            defrag_threshold: 10,
            max_alloc_size,
            alignment,
        }
    }

    /// Create a new virtual address space (page)
    fn create_virtual_space<Storage: VirtualStorage>(
        &mut self,
        page_size: u64,
        storage: &mut Storage,
    ) -> Result<StorageId, IoError> {
        // Reserve a virtual address space
        let storage_handle = storage.reserve(page_size, 0)?;
        let space_id = storage_handle.id;

        let virtual_space = VirtualAddressSpace::new(storage_handle, page_size);
        self.virtual_spaces.insert(space_id, virtual_space);
        self.storage_index.insert(space_id,  page_size);
        self.ring.push_page(space_id);

        Ok(space_id)
    }

    /// Create a new virtual slice within a virtual address space
    fn create_slice_in_space(
        &mut self,
        space_id: StorageId,
        offset: u64,
        size: u64,
    ) -> VirtualSlice {
        assert_eq!(offset % self.alignment, 0, "Slice offset must be aligned");

        let padding = calculate_padding(size, self.alignment);
        let handle = SliceHandle::new();

        VirtualSlice::new(space_id, offset, size, padding, handle)
    }

    /// Find a free slice that can accommodate the requested size
    fn find_free_slice(&self, required_size: u64) -> Option<(SliceId, bool)> {
        for slice in self.slices.values() {
            if !slice.is_free() {
                continue;
            }

            let slice_size = slice.storage_size();

            // Exact match
            if slice_size == required_size {
                return Some((slice.id(), false));
            }

            // Slice is larger, needs splitting
            if slice_size > required_size && slice_size - required_size >= self.alignment {
                return Some((slice.id(), true));
            }
        }

        None
    }

    /// Merge adjacent free slices within the same virtual address space
    fn merge_adjacent_slices<Storage: VirtualStorage>(
        &mut self,
        storage: &mut Storage,
    ) -> Result<(), IoError> {
        let mut merged_any = true;

        while merged_any {
            merged_any = false;

            // Group slices by virtual address space
            let mut slices_by_space: HashMap<StorageId, Vec<SliceId>> = HashMap::new();

            for slice in self.slices.values() {
                if slice.is_free() {
                    slices_by_space
                        .entry(slice.virtual_storage_id)
                        .or_default()
                        .push(slice.id());
                }
            }

            // Try to merge adjacent slices within each space
            for (space_id, slice_ids) in slices_by_space {
                for &slice_id in &slice_ids {
                    if !self.slices.contains_key(&slice_id) {
                        continue; // Already merged
                    }

                    let slice = self.slices.get(&slice_id).unwrap();
                    let slice_end = slice.virtual_offset + slice.effective_size();

                    // Find adjacent slice
                    if let Some(&adjacent_id) = slice_ids.iter().find(|&&id| {
                        if let Some(other) = self.slices.get(&id) {
                            other.virtual_offset == slice_end && other.is_free()
                        } else {
                            false
                        }
                    }) {
                        // Merge the slices
                        if let (Some(mut first), Some(second)) = (
                            self.slices.remove(&slice_id),
                            self.slices.remove(&adjacent_id),
                        ) {
                            // Unmap both if mapped
                            if first.mapped {
                                first.unmap(storage);
                            }
                            if second.mapped {
                                second.unmap(storage);
                            }

                            // Merge sizes
                            first.size += second.effective_size();
                            first.padding = 0;

                            // Update virtual space
                            if let Some(space) = self.virtual_spaces.get_mut(&space_id) {
                                space.remove_slice(second.virtual_offset);
                            }

                            self.slices.insert(slice_id, first);
                            merged_any = true;
                            break;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn defragment<Storage: VirtualStorage>(
        &mut self,
        storage: &mut Storage,
    ) -> Result<(), IoError> {
        // First, merge adjacent free slices within pages
        self.merge_adjacent_slices(storage)?;

        // Collect all free slices with their physical handles
        let mut free_slices: Vec<(SliceId, Option<StorageHandle>)> = self
            .slices
            .values()
            .filter(|slice| slice.is_free())
            .map(|slice| (slice.id(), slice.physical_handle.clone()))
            .collect();

        if free_slices.is_empty() {
            return Ok(());
        }

        // Build adjacency chains using VirtualStorage.are_adjacent()
        let mut adjacency_chains = Vec::new();
        let mut processed = HashSet::new();

        for &(slice_id, ref physical_handle) in &free_slices {
            if processed.contains(&slice_id) {
                continue;
            }

            let mut current_chain = vec![slice_id];
            processed.insert(slice_id);

            // Find all slices adjacent to any slice in the current chain
            let mut found_adjacent = true;
            while found_adjacent {
                found_adjacent = false;

                for &(other_slice_id, ref other_handle) in &free_slices {
                    if processed.contains(&other_slice_id) {
                        continue;
                    }

                    // Check if this slice is adjacent to any slice in the current chain
                    for &chain_slice_id in &current_chain {
                        if let (Some(chain_slice), Some(chain_handle), Some(other_handle)) = (
                            self.slices.get(&chain_slice_id),
                            chain_slice.physical_handle.as_ref(),
                            other_handle.as_ref(),
                        ) {
                            if storage.are_adjacent(chain_handle, other_handle) {
                                current_chain.push(other_slice_id);
                                processed.insert(other_slice_id);
                                found_adjacent = true;
                                break;
                            }
                        }
                    }

                    if found_adjacent {
                        break;
                    }
                }
            }

            adjacency_chains.push(current_chain);
        }

        // Process each adjacency chain
        for chain in adjacency_chains {
            if chain.len() <= 1 {
                continue; // No need to consolidate single slices
            }

            // Calculate total size of the chain
            let total_size: u64 = chain
                .iter()
                .filter_map(|&id| self.slices.get(&id))
                .map(|slice| slice.effective_size())
                .sum();

            if total_size < self.alignment {
                continue;
            }

            // Find the slice with the lowest virtual address to serve as the base
            let base_slice_id = chain
                .iter()
                .min_by_key(|&&id| {
                    self.slices.get(&id)
                        .map(|s| (s.virtual_storage_id, s.virtual_offset))
                        .unwrap_or((StorageId::new(), u64::MAX))
                })
                .copied()
                .unwrap();

            // Unmap and remove all slices in the chain except the base
            for &slice_id in &chain {
                if slice_id == base_slice_id {
                    continue;
                }

                if let Some(mut slice) = self.slices.remove(&slice_id) {
                    if slice.mapped {
                        slice.unmap(storage);
                    }

                    // Remove from virtual space
                    if let Some(space) = self.virtual_spaces.get_mut(&slice.virtual_storage_id) {
                        space.remove_slice(slice.virtual_offset);
                    }
                }
            }

            // Expand the base slice to cover the entire consolidated region
            if let Some(base_slice) = self.slices.get_mut(&base_slice_id) {
                // Unmap the base slice if it was mapped
                if base_slice.mapped {
                    base_slice.unmap(storage);
                }

                // Create a new virtual address space for the consolidated region
                let new_space_id = self.create_virtual_space(storage)?;
                let consolidated_slice = self.create_slice_in_space(new_space_id, 0, total_size);
                let consolidated_slice_id = consolidated_slice.id();

                // Remove the old base slice
                if let Some(space) = self.virtual_spaces.get_mut(&base_slice.virtual_storage_id) {
                    space.remove_slice(base_slice.virtual_offset);
                }

                // Replace base slice with consolidated slice
                self.slices.remove(&base_slice_id);
                self.slices.insert(base_slice_id, consolidated_slice); // Keep same ID for consistency

                // Add to new virtual space
                if let Some(space) = self.virtual_spaces.get_mut(&new_space_id) {
                    space.insert_slice(0, base_slice_id);
                }
            }
        }

        // Clean up empty virtual spaces
        let empty_spaces: Vec<StorageId> = self
            .virtual_spaces
            .iter()
            .filter(|(_, space)| space.is_empty())
            .map(|(id, _)| *id)
            .collect();

        for space_id in empty_spaces {
            if let Some(_) = self.virtual_spaces.remove(&space_id) {
                storage.release(space_id);
            }
        }

        Ok(())
    }
}

impl VirtualMemoryPool for DefragmentableMemoryPool {
    fn max_alloc_size(&self) -> u64 {
        self.max_alloc_size
    }

    fn get(&self, binding: &SliceBinding) -> Option<&StorageHandle> {
        self.slices
            .get(binding.id())
            .and_then(|slice| slice.physical_handle.as_ref())
    }

    fn try_reserve(&mut self, size: u64) -> Option<SliceHandle> {
        // Note: This method doesn't have access to storage, so it can't perform splits
        // that require mapping operations. It should be used carefully or modified
        // to accept storage parameter.
        if size == 0 || size > self.max_alloc_size {
            return None;
        }

        // Only look for exact matches since we can't split without storage access
        for slice in self.slices.values() {
            if slice.is_free() && slice.storage_size() == size {
                return Some(slice.handle.clone());
            }
        }

        None
    }

    fn alloc<Storage: VirtualStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
    ) -> Result<SliceHandle, IoError> {
        if size == 0 {
            return Err(IoError::Unknown("Cannot allocate zero-sized memory".to_string()));
        }

        if size > self.max_alloc_size {
            return Err(IoError::BufferTooBig(size as usize));
        }

        // Try to reuse existing free slice
        if let Some((slice_id, needs_split)) = self.find_free_slice(size) {
            if needs_split {
                // Split the slice with proper mapping handling
                if let Some(slice) = self.slices.get_mut(&slice_id) {
                    match slice.split(size, self.alignment, storage) {
                        Ok(Some(new_slice)) => {
                            let new_slice_id = new_slice.id();
                            let new_slice_offset = new_slice.virtual_offset;
                            let space_id = new_slice.virtual_storage_id;

                            // Add new slice to virtual space
                            if let Some(space) = self.virtual_spaces.get_mut(&space_id) {
                                space.insert_slice(new_slice_offset, new_slice_id);
                            }

                            self.slices.insert(new_slice_id, new_slice);
                        }
                        Ok(None) => {
                            // Split wasn't possible, slice was adjusted with padding
                        }
                        Err(e) => {
                            return Err(e);
                        }
                    }
                }
            }

            // Map the slice to physical memory if not already mapped
            if let Some(slice) = self.slices.get_mut(&slice_id) {
                if !slice.is_mapped() {
                    slice.map(storage, &self.virtual_spaces)?;
                }
                return Ok(slice.handle.clone());
            }
        }

        self.alloc_count += 1;

        // Check if we should defragment
        if self.alloc_count >= self.defrag_threshold {
            self.defragment(storage)?;
            self.alloc_count = 0;
        }

        // Create new virtual space if needed
        let space_id = self.create_virtual_space(storage)?;

        // Create slice in the new space
        let mut new_slice = self.create_slice_in_space(space_id, 0, size);
        let handle = new_slice.handle.clone();
        let slice_id = new_slice.id();

        // Map to physical memory
        new_slice.map(storage, &self.virtual_spaces)?;

        // Add to virtual space
        if let Some(space) = self.virtual_spaces.get_mut(&space_id) {
            space.insert_slice(0, slice_id);
        }

        // Create remaining free slice if there's leftover space in the page
        let effective_size = new_slice.effective_size();
        if effective_size < self.page_size {
            let remaining_size = self.page_size - effective_size;
            let remaining_slice = self.create_slice_in_space(space_id, effective_size, remaining_size);
            let remaining_id = remaining_slice.id();

            if let Some(space) = self.virtual_spaces.get_mut(&space_id) {
                space.insert_slice(effective_size, remaining_id);
            }

            self.slices.insert(remaining_id, remaining_slice);
        }

        self.slices.insert(slice_id, new_slice);

        Ok(handle)
    }

    fn get_memory_usage(&self) -> MemoryUsage {
        let used_slices: Vec<_> = self
            .slices
            .values()
            .filter(|slice| !slice.is_free())
            .collect();

        MemoryUsage {
            number_allocs: used_slices.len() as u64,
            bytes_in_use: used_slices.iter().map(|s| s.storage_size()).sum(),
            bytes_padding: used_slices.iter().map(|s| s.padding).sum(),
            bytes_reserved: (self.virtual_spaces.len() as u64) * self.page_size,
        }
    }

    fn cleanup<Storage: VirtualStorage>(
        &mut self,
        storage: &mut Storage,
        _alloc_nr: u64,
        explicit: bool,
    ) {
        if explicit {
            // Unmap and remove all free slices
            self.slices.retain(|_, slice| {
                if slice.is_free() {
                    if slice.mapped {
                        slice.unmap(storage);
                    }
                    false
                } else {
                    true
                }
            });

            // Clean up empty virtual spaces
            let empty_spaces: Vec<StorageId> = self
                .virtual_spaces
                .iter()
                .filter(|(_, space)| space.is_empty())
                .map(|(id, _)| *id)
                .collect();

            for space_id in empty_spaces {
                self.virtual_spaces.remove(&space_id);
                storage.release(space_id);
            }

            storage.cleanup();
        }
    }
}
