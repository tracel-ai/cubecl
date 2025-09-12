use crate::server::IoError;
use crate::storage::{StorageHandle, StorageId, StorageUtilization};

// Enum with possible states for a virtual block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VirtualHandleState {
    Unmapped, // Virtual space reserved, no physical memory
    Mapped,   // Physical memory mapped, available for allocation
}

/// A contiguous block of virtual memory that can be mapped to physical handles.
#[derive(Debug)]
pub struct VirtualHandle {
    state: VirtualHandleState,
    handle: StorageHandle,
}

impl VirtualHandle {
    pub fn new(id: StorageId, utilization: StorageUtilization) -> Self {
        Self {
            state: VirtualHandleState::Unmapped,
            handle: StorageHandle::new(id, utilization),
        }
    }

    pub fn size(&self) -> u64 {
        self.handle.size()
    }

    pub fn id(&self) -> StorageId {
        self.handle.id
    }

    pub fn offset(&self) -> u64 {
        self.handle.offset()
    }

    /// Map the block to mark it as mapped
    pub fn set_mapped(&mut self) {
        self.state = VirtualHandleState::Mapped;
    }

    pub fn is_mapped(&self) -> bool {
        matches!(self.state, VirtualHandleState::Mapped)
    }

    pub fn is_unmapped(&self) -> bool {
        matches!(self.state, VirtualHandleState::Unmapped)
    }

    /// Marks the block as unmapped from physical memory.
    pub fn set_unmapped(&mut self) {
        self.state = VirtualHandleState::Unmapped;
    }
}
pub trait VirtualStorage: Send {
    type Resource: Send;

    fn alignment(&self) -> usize;
    fn get(&mut self, handle: &VirtualHandle) -> Self::Resource;

    // Allocates a enough blocks to match the specified size.
    // Does not perform any mapping nor unmapping.
    // The memory management module should be in charge of this using the underlying [`map`] method.
    // Returns an unmapped virtual memory handle
    fn alloc(&mut self, size: u64) -> Result<VirtualHandle, IoError>;

    // Will set all the physical handles of this virtual block to the deallocation list.
    // The block will be unmapped if it has not yet been.
    fn dealloc(&mut self, id: StorageId);

    // Will flush all pending deallocations, releasing physical memory.
    fn flush(&mut self);

    // Maps a physical memory range of an specified size to a set of contiguous virtual memory addresses. Returns a mapped handle.
    fn map(&mut self, handle: &mut VirtualHandle) -> Result<(), IoError>;

    // Just unmaps a block of virtual memory, allowing for future reuse of its underlying physical handles.
    fn unmap(&mut self, id: StorageId);
}
