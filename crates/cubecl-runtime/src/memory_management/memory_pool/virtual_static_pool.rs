use super::{SliceBinding, SliceHandle, SliceId, calculate_padding};
use crate::memory_management::MemoryUsage;
use crate::server::IoError;
use crate::storage::{
    PhysicalStorageHandle,  StorageHandle,
    VirtualSpaceId, VirtualStorage,
};
use alloc::vec::Vec;
use hashbrown::HashMap;
use crate::memory_management::memory_pool::VirtualMemoryPool;


/// A virtual memory pool that leverages automatic merge/split behavior.
/// The main advantage of using virtual memory is that it allows to merge/split pages without
/// any additional overhead other than releasing the mapped address space back to the device
/// and unmapping the mapped physical blocks.
/// It is preferred that blocks are fixed size for this, ideally of equal size as that of the
/// minimum allocation granularity of the target device.
/// This allows [`VirtualSlices`] to be split and merged back automatically once they are freed with no overhead (This is why you do not see any) split_block or try_merge functions here.
/// The workflow is the following:
/// [`VirtualSlices`] represent mapped virtual memory regions.
/// When a [`VirtualSlice`] is freed, its physical blocks are unmapped and virtual memory is released.
/// The physical blocks go to the [`free_list`] so that they can be remapped later to build a new [`VirtualSlice`].
/// As we are working with virtual memory, virtual address spaces stay always contiguous, no matter the physical blocks are not contiguous in physical memory.
pub(crate) struct VirtualStaticPool{


    // Physical block tracking (never freed, only reused)
    physical_blocks: Vec<PhysicalStorageHandle>,
    free_block_indices: Vec<usize>, // Indices into physical_blocks that are free

    // Active allocations (each allocation gets its own virtual space)
    active_slices: HashMap<SliceId, VirtualSlice>,

    // Pool configuration
    block_size: u64, // Fixed block size is preferred when working with virtual memory
    max_alloc_size: u64,
    alignment: u64, // minimum granularity.
}

/// Represents an active allocation in virtual memory.
/// Each slice corresponds to one virtual address space mapping multiple physical blocks.
#[derive(Debug)]
struct VirtualSlice {
    /// Handle for external reference
    handle: SliceHandle,
    /// Storage handle for the mapped virtual space
    storage_handle: StorageHandle,
    /// Virtual space ID
    virtual_space_id: VirtualSpaceId,
    /// Physical blocks backing this allocation (indices into the physical block Vec)
    physical_block_indices: Vec<usize>,
    /// User-requested size (may be less than total virtual space)
    requested_size: u64,
    /// Padding added for alignment
    padding: u64, // We need to ensure total slice size is aligned to the pool's block_size.
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

    fn is_free(&self) -> bool {
        self.handle.is_free()
    }

    // Helper to get block size
    fn get_block_size(&self) -> u64 {
        let total_size = self.requested_size + self.padding;
        total_size.div_ceil(self.physical_block_indices.len() as u64)
    }
}

impl VirtualStaticPool {
    pub(crate) fn new<Storage: VirtualStorage>(storage: &Storage, block_size: u64, max_alloc_size: u64) -> Self {
        let alignment = storage.alignment() as u64;
        let block_size = block_size.next_multiple_of(alignment);

        Self {
            physical_blocks: Vec::new(),
            free_block_indices: Vec::new(),
            active_slices: HashMap::new(),
            block_size,
            max_alloc_size,
            alignment,
        }
    }

    /// Get or allocate physical blocks
    fn get_physical_blocks<Storage: VirtualStorage>(&mut self, storage: &mut Storage, count: usize) -> Result<Vec<usize>, IoError> {
        let mut block_indices = Vec::new();

        // First try to reuse existing free blocks
        for _ in 0..count {
            if let Some(free_index) = self.free_block_indices.pop() {
                block_indices.push(free_index);
            } else {
                // Allocate new physical block.
                let physical_handle = storage.alloc(self.block_size)?;
                self.physical_blocks.push(physical_handle);
                block_indices.push(self.physical_blocks.len() - 1);
            }
        }

        Ok(block_indices)
    }

    /// Create a new allocation by mapping physical blocks to a fresh virtual space
    /// Blocks can be either freshly allocated or reused from the freelist.
    /// The logic to determine where do blocks come from is determined by the function
    /// [`get_physical_blocks`]
    fn create_allocation<Storage: VirtualStorage>(&mut self, storage: &mut Storage, size: u64) -> Result<VirtualSlice, IoError> {

        let padding = calculate_padding(size, self.alignment);
        let total_size = size + padding;
        let blocks_needed = total_size.div_ceil(self.block_size) as usize;

        // Get physical blocks (reuse existing or allocate new)
        let physical_block_indices = self.get_physical_blocks(storage, blocks_needed)?;

        // Create fresh virtual address space
        let virtual_size = blocks_needed as u64 * self.block_size;
        let virtual_handle = storage.try_reserve(virtual_size as usize)?;
        let virtual_space_id = virtual_handle.id();

        // Collect physical handles for mapping
        let physical_handles: Vec<_> = physical_block_indices
            .iter()
            .map(|&index| self.physical_blocks[index].clone())
            .collect();

        // Map all physical blocks to the virtual space (1:N mapping)
        let storage_handle = storage.try_map(virtual_handle, physical_handles)?;

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
    /// Here, by unmapping and then releasing the virtual space,
    /// we make physical blocks available for reuse.
    fn free_allocation<Storage: VirtualStorage>(&mut self,storage: &mut Storage,slice: VirtualSlice) -> Result<(), IoError> {
        // Unmap the virtual-to-physical mapping
        storage.try_unmap(slice.storage_handle.id)?;

        // Release the virtual address space back to the system
        storage.try_release(slice.virtual_space_id)?;

        // Mark physical blocks as free for reuse
        self.free_block_indices
            .extend(&slice.physical_block_indices);

        Ok(())
    }
}

impl VirtualMemoryPool for VirtualStaticPool {
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
    /// This is a statically sized virtual pool.
    /// Although Virtual Slices are dynamically sized, physical blocks are always the same size.
    /// Idea for the future is to add a [`SlicedVirtualPool`] that allows to use dynamically-sized physical blocks.
    /// For now, I am following the pattern in `StaticPool`.
    fn try_reserve<Storage: VirtualStorage>(&mut self, _storage: &mut Storage, _size: u64) -> Option<SliceHandle> {
       None
    }

    /// Create new allocation with fresh virtual address space
    fn alloc<Storage: VirtualStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
    ) -> Result<SliceHandle, IoError> {
        if size > self.max_alloc_size {
            return Err(IoError::BufferTooBig(size as usize));
        }

        let slice = self.create_allocation(storage, size)?;
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

    /// Cleanup by freeing virtual address spaces
    ///
    /// When we free allocations, their physical blocks become available for
    /// any future allocation of any size. This is the "automatic merge" -
    /// there's no need to track adjacency or explicitly merge ranges.
    fn cleanup<Storage: VirtualStorage>(
        &mut self,
        storage: &mut Storage,
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
                    self.free_allocation(storage, slice).ok();
                }
            }

          }
    }
}
