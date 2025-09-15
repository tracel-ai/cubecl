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

/// A simplified virtual memory pool that leverages automatic merge/split behavior.
///
/// Key insights:
/// - Merge happens automatically when we free slices: unmap + release virtual space
/// - Split happens automatically when we allocate: just create new virtual spaces
/// - Minimum allocation granularity is block_size (physical blocks are never split)
/// - Physical blocks are never freed, only reused
pub(crate) struct VirtualStaticPool<Storage: VirtualStorage> {
    // Virtual memory management
    storage: Storage,

    // Physical block tracking (never freed, only reused)
    physical_blocks: Vec<PhysicalStorageHandle>,
    free_block_indices: Vec<usize>, // Indices into physical_blocks that are free

    // Active allocations (each allocation gets its own virtual space)
    active_slices: HashMap<SliceId, VirtualSlice>,

    // Pool configuration
    block_size: u64,
    max_alloc_size: u64,
    alignment: u64,
}

/// Represents an active allocation in virtual memory.
/// Each slice corresponds to one virtual address space mapping multiple physical blocks.
#[derive(Debug)]
struct VirtualSlice {
    /// Handle for external reference
    handle: SliceHandle,
    /// Storage handle for the mapped virtual space
    storage_handle: StorageHandle,
    /// Virtual space ID (for cleanup)
    virtual_space_id: VirtualSpaceId,
    /// Physical blocks backing this allocation (indices into physical_blocks)
    physical_block_indices: Vec<usize>,
    /// User-requested size (may be less than total virtual space)
    requested_size: u64,
    /// Padding added for alignment
    padding: u64,
}

impl VirtualSlice {
    fn new(
        handle: SliceHandle,
        storage_handle: StorageHandle,
        virtual_space_id: VirtualSpaceId,
        physical_block_indices: Vec<usize>,
        requested_size: u64,
        padding: u64,
    ) -> Self {
        Self {
            handle,
            storage_handle,
            virtual_space_id,
            physical_block_indices,
            requested_size,
            padding,
        }
    }

    fn is_free(&self) -> bool {
        self.handle.is_free()
    }

    fn id(&self) -> SliceId {
        *self.handle.id()
    }

    fn storage_handle(&self) -> &StorageHandle {
        &self.storage_handle
    }

    /// Total virtual size (always multiple of block_size)
    fn virtual_size(&self) -> u64 {
        self.requested_size + self.padding
    }

    // Helper to get block size
    fn get_block_size(&self) -> u64 {

        let total_size = self.requested_size + self.padding;
        total_size.div_ceil(self.physical_block_indices.len() as u64)
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
            free_block_indices: Vec::new(),
            active_slices: HashMap::new(),
            block_size,
            max_alloc_size,
            alignment,
        }
    }

    /// Get or allocate physical blocks
    fn get_physical_blocks(&mut self, count: usize) -> Result<Vec<usize>, IoError> {
        let mut block_indices = Vec::new();

        // First try to reuse existing free blocks
        for _ in 0..count {
            if let Some(free_index) = self.free_block_indices.pop() {
                block_indices.push(free_index);
            } else {
                // Allocate new physical block
                let physical_handle = self.storage.alloc(self.block_size)?;
                self.physical_blocks.push(physical_handle);
                block_indices.push(self.physical_blocks.len() - 1);
            }
        }

        Ok(block_indices)
    }

    /// Create a new allocation by mapping physical blocks to a fresh virtual space
    ///
    /// This is where the "automatic split" happens - each allocation gets its own
    /// virtual address space, so there's no need to explicitly split existing ones.
    fn create_allocation(&mut self, size: u64) -> Result<VirtualSlice, IoError> {
        let padding = calculate_padding(size, self.alignment);
        let total_size = size + padding;
        let blocks_needed = ((total_size + self.block_size - 1) / self.block_size) as usize;

        // Get physical blocks (reuse existing or allocate new)
        let physical_block_indices = self.get_physical_blocks(blocks_needed)?;

        // Create fresh virtual address space
        let virtual_size = blocks_needed as u64 * self.block_size;
        let virtual_handle = self.storage.try_reserve(virtual_size as usize)?;
        let virtual_space_id = virtual_handle.id();

        // Collect physical handles for mapping
        let physical_handles: Vec<_> = physical_block_indices
            .iter()
            .map(|&index| self.physical_blocks[index].clone())
            .collect();

        // Map all physical blocks to the virtual space (1:N mapping)
        let storage_handle = self.storage.try_map(virtual_handle, physical_handles)?;

        // Create slice
        let handle = SliceHandle::new();
        let slice = VirtualSlice::new(
            handle,
            storage_handle,
            virtual_space_id,
            physical_block_indices,
            size,
            padding,
        );

        Ok(slice)
    }

    /// Free an allocation by unmapping and releasing virtual space
    ///
    /// This is where "automatic merge" happens - by releasing the virtual space,
    /// we make the physical blocks available for any future allocation regardless
    /// of size. There's no fragmentation in virtual space since each allocation
    /// gets a fresh virtual address range.
    fn free_allocation(&mut self, slice: VirtualSlice) -> Result<(), IoError> {
        // Unmap the virtual-to-physical mapping
        self.storage.try_unmap(slice.storage_handle.id)?;

        // Release the virtual address space back to the system
        self.storage.try_release(slice.virtual_space_id)?;

        // Mark physical blocks as free for reuse
        // This is the key: physical blocks become available for ANY future allocation
        self.free_block_indices.extend(&slice.physical_block_indices);

        Ok(())
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

    /// Try to reuse existing free slices or create new allocation
    ///
    /// Note: Since minimum granularity is block_size, we can only reuse slices
    /// that are exactly the right size or larger by full block increments.
    fn try_reserve(&mut self, size: u64) -> Option<SliceHandle> {
        let padding = calculate_padding(size, self.alignment);
        let total_size = size + padding;
        let blocks_needed = ((total_size + self.block_size - 1) / self.block_size) as usize;
        let required_virtual_size = blocks_needed as u64 * self.block_size;

        // Look for a free slice that's exactly the right size
        // (We can't split since minimum granularity is block_size)
        for (slice_id, slice) in &self.active_slices {
            if slice.is_free() && slice.virtual_size() == required_virtual_size {
                return Some(slice.handle.clone());
            }
        }

        // No exact match found - would need to allocate new
        None
    }

    /// Create new allocation with fresh virtual address space
    ///
    /// This demonstrates the "automatic split" behavior - instead of splitting
    /// existing allocations, we just create new ones. Each gets its own virtual
    /// address space, so there's no fragmentation.
    fn alloc<ComputeStorage: crate::storage::ComputeStorage>(
        &mut self,
        _storage: &mut ComputeStorage, // Not used, we use VirtualStorage instead
        size: u64,
    ) -> Result<SliceHandle, IoError> {
        if size > self.max_alloc_size {
            return Err(IoError::BufferTooBig(size as usize));
        }

        let slice = self.create_allocation(size)?;
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

    /// Cleanup by freeing virtual address spaces (automatic merge behavior)
    ///
    /// When we free allocations, their physical blocks become available for
    /// any future allocation of any size. This is the "automatic merge" -
    /// there's no need to track adjacency or explicitly merge ranges.
    fn cleanup<ComputeStorage: crate::storage::ComputeStorage>(
        &mut self,
        _storage: &mut ComputeStorage, // Not used
        _alloc_nr: u64,
        explicit: bool,
    ) {
        if explicit {
            let mut slices_to_free = Vec::new();

            // Collect free slices
            for (slice_id, slice) in &self.active_slices {
                if slice.is_free() {
                    slices_to_free.push(*slice_id);
                }
            }

            // Free them (automatic merge happens here)
            for slice_id in slices_to_free {
                if let Some(slice) = self.active_slices.remove(&slice_id) {
                    // This is where the magic happens: unmapping + releasing virtual space
                    // makes all physical blocks available for any future allocation
                    self.free_allocation(slice).ok();
                }
            }

            self.storage.flush();
        }
    }
}
