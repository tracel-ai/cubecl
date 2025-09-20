use crate::server::IoError;
use crate::storage::{StorageHandle, StorageId};

pub trait VirtualStorage {
    type VirtualAddress: Send; // Should be defined by the backend
    type Resource: Send;

    /// Retrieves the minimum allocation granularity of this storage. All physical and virtual allocations should be aligned.
    fn granularity(&self) -> usize;
    // Retrieves the size of a physical block in this virtual storage.
    fn physical_block_size(&self) -> usize;

    /// Retrieves an active mapping
    fn get(&mut self, handle: &StorageHandle) -> Self::Resource;

    /// Reserves an address space of a given size. Padding should be automatically added to meet the granularity requirements.
    fn reserve(&mut self, size: usize) -> Result<Self::VirtualAddress, IoError>;

    /// Allocates a single physical block of
    fn alloc_physical(&mut self) -> Result<(), IoError>;

    /// Map a range of virtual addresses to physical memory.
    fn map(
        &mut self,
        start_address: Self::VirtualAddress,
        size: usize,
    ) -> Result<StorageHandle, IoError>;

    /// Unmap that range.
    fn unmap(&mut self, id: StorageId);

    /// Releases all address spaces and cleans up all physical memory.
    fn cleanup(&mut self);
}
