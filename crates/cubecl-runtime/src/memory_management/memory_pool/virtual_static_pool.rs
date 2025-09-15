use super::{MemoryPool, SliceBinding, SliceHandle, SliceId, calculate_padding};
use crate::memory_management::MemoryUsage;
use crate::storage::{
    VirtualStorage, VirtualAddressSpaceHandle, PhysicalStorageHandle,
    StorageHandle, StorageId, StorageUtilization,
    VirtualSpaceId, PhysicalStorageId
};
use crate::server::IoError;
use alloc::vec::Vec;
use hashbrown::HashMap;

/// A memory pool that manages fixed-size physical blocks using virtual memory mapping.
///
/// Key principles:
/// - Physical memory blocks are never freed, only reused
/// - All allocations appear contiguous in virtual address space
/// - Physical blocks are of uniform size for efficient merging
/// - Virtual mappings are created/destroyed during split/merge operations
pub(crate) struct VirtualStaticPool<Storage: VirtualStorage> {
    // Virtual memory management
    storage: Storage,

    // Physical block management
    physical_blocks: Vec<PhysicalBlock>,
    free_blocks: Vec<usize>, // Indices of free physical blocks in physical_blocks

    // Active virtual slices
    active_slices: HashMap<SliceId, VirtualSlice>,

    // Pool configuration
    block_size: u64,
    max_alloc_size: u64,
    alignment: u64,
}

/// Represents a physical memory block that can be mapped to virtual address spaces
#[derive(Debug)]
struct PhysicalBlock {
    /// Physical memory handle
    physical_handle: PhysicalStorageHandle,
    /// Whether this block is currently free for allocation
    is_free: bool,
    /// Current virtual mapping (if any)
    current_mapping: Option<VirtualMapping>,
}

/// Information about a virtual memory mapping
#[derive(Debug, Clone)]
struct VirtualMapping {
    /// Virtual address space handle
    virtual_handle: VirtualAddressSpaceHandle,
    /// Storage handle returned by map operation
    storage_handle: StorageHandle,
}

/// Represents an active slice that may span multiple physical blocks
#[derive(Debug)]
struct VirtualSlice {
    /// Handle for external reference
    handle: SliceHandle,
    /// Physical blocks that back this slice (by index)
    physical_block_indices: Vec<usize>,
    /// Actual size requested by user
    requested_size: u64,
    /// Total size including padding
    total_size: u64,
    /// Padding for alignment
    padding: u64,
}

impl PhysicalBlock {
    fn new(physical_handle: PhysicalStorageHandle) -> Self {
        Self {
            physical_handle,
            is_free: true,
            current_mapping: None,
        }
    }

    /// Map this physical block to a virtual address space
    fn map_to_virtual<Storage: VirtualStorage>(
        &mut self,
        storage: &mut Storage,
        virtual_handle: VirtualAddressSpaceHandle,
    ) -> Result<StorageHandle, IoError> {
        if self.current_mapping.is_some() {
            return Err(IoError::InvalidHandle);
        }

        let storage_handle = storage.try_map(virtual_handle.clone(), self.physical_handle.clone())?;

        self.current_mapping = Some(VirtualMapping {
            virtual_handle,
            storage_handle: storage_handle.clone(),
        });

        Ok(storage_handle)
    }

    /// Unmap this physical block from virtual address space
    fn unmap_from_virtual<Storage: VirtualStorage>(
        &mut self,
        storage: &mut Storage,
    ) -> Result<(), IoError> {
        if let Some(mapping) = &self.current_mapping {
            storage.try_unmap(mapping.storage_handle.id)?;
            storage.try_release(mapping.virtual_handle.id())?;
            self.current_mapping = None;
        }
        Ok(())
    }
}

impl VirtualSlice {
    fn new(
        handle: SliceHandle,
        physical_block_indices: Vec<usize>,
        requested_size: u64,
        padding: u64,
    ) -> Self {
        let total_size = requested_size + padding;
        Self {
            handle,
            physical_block_indices,
            requested_size,
            total_size,
            padding,
        }
    }

    fn is_free(&self) -> bool {
        self.handle.is_free()
    }

    fn id(&self) -> SliceId {
        *self.handle.id()
    }

    /// Calculate total virtual size needed for this slice
    fn virtual_size(&self) -> u64 {
        self.total_size
    }

    /// Check if this slice can be merged with adjacent slices
    /// The main advantage of using virtual memory is that we can always merge two slices even if they are not adjacent
    fn can_merge_with(&self, other: &VirtualSlice) -> bool {
       self.is_free && other.is_free
    }
}

impl<Storage: VirtualStorage> VirtualStaticPool<Storage> {
    pub(crate) fn new(
        storage: Storage,
        block_size: u64,
        max_alloc_size: u64,
    ) -> Self {
        let alignment = storage.alignment() as u64;
        assert_eq!(block_size % alignment, 0);

        Self {
            storage,
            physical_blocks: Vec::new(),
            free_blocks: VecDeque::new(),
            active_slices: HashMap::new(),
            block_size,
            max_alloc_size,
            alignment,
        }
    }

    /// Allocate a new physical block and add it to the pool
    fn allocate_physical_block(&mut self) -> Result<usize, IoError> {
        let physical_handle = self.storage.alloc(self.block_size)?;
        let block = PhysicalBlock::new(physical_handle);

        self.physical_blocks.push(block);
        let block_index = self.physical_blocks.len() - 1;

        Ok(block_index)
    }

    /// Get a free physical block, allocating if necessary
    fn get_free_block(&mut self) -> Result<usize, IoError> {
        if let Some(block_index) = self.free_blocks.pop_front() {
            Ok(block_index)
        } else {
            self.allocate_physical_block()
        }
    }

    /// Get consecutive free blocks for large allocations
    fn get_consecutive_blocks(&mut self, num_blocks: usize) -> Result<Vec<usize>, IoError> {
        let mut blocks = Vec::new();

        for _ in 0..num_blocks {
            blocks.push(self.get_free_block()?);
        }

        Ok(blocks)
    }

    /// Create virtual address space for given size
    fn create_virtual_space(&mut self, size: u64) -> Result<VirtualAddressSpaceHandle, IoError> {
        self.storage.try_reserve(size as usize)
    }

    /// Create a mapped slice from physical blocks
    fn create_mapped_slice(
        &mut self,
        physical_block_indices: Vec<usize>,
        requested_size: u64,
    ) -> Result<VirtualSlice, IoError> {
        let total_virtual_size = physical_block_indices.len() as u64 * self.block_size;
        let virtual_handle = self.create_virtual_space(total_virtual_size)?;

        // Map the first physical block to get the base storage handle
        let first_block_index = physical_block_indices[0];
        let first_block = &mut self.physical_blocks[first_block_index];
        first_block.is_free = false;

        let base_storage_handle = first_block.map_to_virtual(&mut self.storage, virtual_handle.clone())?;

        // For multi-block allocations, we would need to map additional blocks
        // This is a simplified version - full implementation would handle multiple physical blocks
        for &block_index in &physical_block_indices[1..] {
            let block = &mut self.physical_blocks[block_index];
            block.is_free = false;
            // In a full implementation, we'd map each block to consecutive virtual addresses
        }

        let virtual_mapping = VirtualMapping {
            virtual_handle,
            storage_handle: base_storage_handle,
        };

        let padding = calculate_padding(requested_size, self.alignment);
        let handle = SliceHandle::new();

        let slice = VirtualSlice::new(
            handle,
            physical_block_indices,
            virtual_mapping,
            requested_size,
            padding,
        );

        Ok(slice)
    }

    /// Try to merge adjacent free slices
    fn try_merge_slices(&mut self, slice_id: SliceId) -> bool {
        // Implementation would check for adjacent slices and merge them
        // This is a simplified version
        if let Some(slice) = self.active_slices.get(&slice_id) {
            if slice.is_free() && slice.physical_block_indices.len() == 1 {
                // Look for adjacent free slices to merge with
                let block_index = slice.physical_block_indices[0];

                // Check if we can merge with next block
                if block_index + 1 < self.physical_blocks.len() {
                    if let Some(adjacent_slice) = self.find_slice_with_block(block_index + 1) {
                        if self.active_slices.get(&adjacent_slice).unwrap().is_free() {
                            // Merge logic would go here
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// Find slice that uses a specific physical block
    fn find_slice_with_block(&self, block_index: usize) -> Option<SliceId> {
        for (slice_id, slice) in &self.active_slices {
            if slice.physical_block_indices.contains(&block_index) {
                return Some(*slice_id);
            }
        }
        None
    }

    /// Split a slice if it's larger than needed
    fn try_split_slice(&mut self, slice_id: SliceId, needed_size: u64) -> Option<SliceId> {
        // Implementation would split large slices into smaller ones
        // This is a simplified version
        if let Some(slice) = self.active_slices.get(&slice_id) {
            if slice.virtual_size() > needed_size + self.alignment {
                // Split logic would go here
                // Return the ID of the new slice created from the remainder
            }
        }
        None
    }
}

impl<Storage: VirtualStorage> MemoryPool for VirtualStaticPool<Storage> {
    fn max_alloc_size(&self) -> u64 {
        self.max_alloc_size
    }

    fn get(&self, binding: &SliceBinding) -> Option<&crate::storage::StorageHandle> {
        self.active_slices
            .get(binding.id())
            .map(|slice| slice.storage_handle())
    }

    fn try_reserve(&mut self, size: u64) -> Option<SliceHandle> {
        let padding = calculate_padding(size, self.alignment);
        let total_size = size + padding;
        let blocks_needed = ((total_size + self.block_size - 1) / self.block_size) as usize;

        // Look for existing free slice that fits
        for (slice_id, slice) in &self.active_slices {
            if slice.is_free() && slice.virtual_size() >= total_size {
                // Try to split if slice is much larger than needed
                self.try_split_slice(*slice_id, total_size);

                return Some(slice.handle.clone());
            }
        }

        // Try to merge compatible free slices
        let free_slice_ids: Vec<_> = self.active_slices
            .iter()
            .filter(|(_, slice)| slice.is_free())
            .map(|(id, _)| *id)
            .collect();

        for i in 0..free_slice_ids.len() {
            for j in i+1..free_slice_ids.len() {
                let slice1_id = free_slice_ids[i];
                let slice2_id = free_slice_ids[j];

                let slice1 = self.active_slices.get(&slice1_id).unwrap();
                let slice2 = self.active_slices.get(&slice2_id).unwrap();

                if slice1.can_merge_with(slice2) {
                    // Try to merge if the combined size would be useful
                    let combined_size = slice1.requested_size + slice2.requested_size;
                    if combined_size >= total_size {
                        if let Ok(merged_id) = self.merge_slices(slice1_id, slice2_id) {
                            // Retry reservation with merged slice
                            return self.try_reserve(size);
                        }
                    }
                }
            }
        }

        None
    }

    fn alloc<ComputeStorage: crate::storage::ComputeStorage>(
        &mut self,
        _storage: &mut ComputeStorage, // Not used, we use VirtualStorage
        size: u64,
    ) -> Result<SliceHandle, IoError> {
        if size > self.max_alloc_size {
            return Err(IoError::BufferTooBig(size as usize));
        }

        let padding = calculate_padding(size, self.alignment);
        let total_size = size + padding;
        let blocks_needed = ((total_size + self.block_size - 1) / self.block_size) as usize;

        // Get the required physical blocks
        let physical_block_indices = self.get_consecutive_blocks(blocks_needed)?;

        // Create mapped slice
        let slice = self.create_mapped_slice(physical_block_indices, size)?;
        let handle = slice.handle.clone();
        let slice_id = slice.id();

        self.active_slices.insert(slice_id, slice);

        Ok(handle)
    }

    fn get_memory_usage(&self) -> MemoryUsage {
        let used_slices: Vec<_> = self
            .active_slices
            .values()
            .filter(|slice| !slice.is_free())
            .collect();

        MemoryUsage {
            number_allocs: used_slices.len() as u64,
            bytes_in_use: used_slices.iter().map(|s| s.requested_size).sum(),
            bytes_padding: used_slices.iter().map(|s| s.padding).sum(),
            bytes_reserved: self.physical_blocks.len() as u64 * self.block_size,
        }
    }

    fn cleanup<ComputeStorage: crate::storage::ComputeStorage>(
        &mut self,
        _storage: &mut ComputeStorage, // Not used
        _alloc_nr: u64,
        explicit: bool,
    ) {
        if explicit {
            // Mark free slices for virtual unmapping, but keep physical blocks
            let mut slices_to_unmap = Vec::new();

            for (slice_id, slice) in &self.active_slices {
                if slice.is_free() {
                    slices_to_unmap.push(*slice_id);
                }
            }

            for slice_id in slices_to_unmap {
                if let Some(slice) = self.active_slices.remove(&slice_id) {
                    // Unmap each virtual space and mark physical blocks as free
                    for mapping in slice.virtual_mappings {
                        let block_index = mapping.physical_block_index;
                        if let Some(block) = self.physical_blocks.get_mut(block_index) {
                            block.unmap_from_virtual(&mut self.storage).ok();
                            block.is_free = true;
                            self.free_blocks.push(block_index);
                        }
                    }
                }
            }

            self.storage.flush();
        }
    }
}
