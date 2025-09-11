use crate::storage::{StorageHandle, StorageId};
use cubecl_core::server::IoError;


// Enum with possible states for a virtual block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlockState {
    Unmapped, // Virtual space reserved, no physical memory
    Mapped,   // Physical memory mapped, available for allocation
}

/// A contiguous block of virtual memory that can be mapped to physical handles.
#[derive(Debug)]
pub struct VirtualBlock {
    state: BlockState,
    /// Number of physical handles backing this virtual block
    virtual_size: usize, // Total size of the block (number of mapped physical handles.)
    base_addr: StorageId, // Base virtual address or the block
}


impl VirtualBlock {
    fn from_reserved(base_addr: StorageId, virtual_size: usize) -> Self {
        let num_handles = (virtual_size / handle_size) as usize;
        Self {
            base_addr,
            state: BlockState::Unmapped,
            virtual_size,
        }
    }

    /// Map the block to mark it as mapped
    fn set_mapped(&mut self) {
        self.state = BlockState::Mapped;
    }

    /// Marks the block as unmapped from physical memory.
    fn set_unmapped(&mut self) {
        self.state = BlockState::Unmapped;
    }

    pub fn get_ptr(&self) -> CUdeviceptr {
        self.base_addr
    }


}
pub trait VirtualStorage: Send {
    type Resource: Send;

    fn alignment(&self) -> usize;
    fn get(&mut self, handle: &StorageHandle) -> Self::Resource;

    // Allocates a physical block of the specified size.
    // Does not perform any mapping nor unmapping.
    // The memory management module should be in charge of this.
    // Returns a pointer to the allocated physical memory handle.
    fn alloc(&mut self, size: u64) -> Result<StorageId, IoError>;

    // Will set a physical block to the deallocation list.
    fn dealloc(&mut self, id: StorageId);

    // Will flush all pending deallocations.
    fn flush(&mut self);

    // Maps a physical memory range of an specified size to a set of contiguous virtual memory addresses.
    fn map(&mut self,  size: usize)-> Result<StorageHandle, IoError>;

    // Just unmaps a block of virtual memory, allowing for future reuse of its underlying physical handles.
    fn unmap(&mut self, id: StorageId);
}
