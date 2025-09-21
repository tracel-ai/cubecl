use crate::server::IoError;
use crate::storage::ComputeStorage;
use crate::storage::{StorageHandle, StorageId};

pub trait VirtualStorage: ComputeStorage {
    /// Retrieves the minimum allocation granularity of this storage. All physical and virtual allocations should be aligned.
    fn granularity(&self) -> usize;
    // Retrieves the size of a physical block in this virtual storage.
    fn physical_block_size(&self) -> usize;

    // Split a range of virtual addresses into two
    fn split_range(
        &mut self,
        handle: &mut StorageHandle,
        offset: usize,
    ) -> Result<StorageHandle, IoError>;

    // Merge two ranges of virtual addresses (either contiguous or not) into one.
    fn merge(
        &mut self,
        first_handle: StorageHandle,
        second_handle: StorageHandle,
    ) -> Result<StorageHandle, IoError>;

    // Expand a virtual memory region to a target size.
    // The difference between merge and expand is that the second one does not require the handle to be unmapped
    fn expand(&mut self, handle: &mut StorageHandle, additional_size: u64) -> Result<(), IoError>;

    /// Reserves an address space of a given size. Padding should be automatically added to meet the granularity requirements. The parameter start_addr is the address at which the reservation should start, if applicable.
    fn reserve(&mut self, size: usize, start_addr: u64) -> Result<StorageHandle, IoError>;

    // Releases the virtual address range associated with this handle.
    fn release(&mut self, handle: StorageHandle);

    /// Map a range of virtual addresses to physical memory.
    fn map(&mut self, handle: &mut StorageHandle) -> Result<(), IoError>;

    /// Unmap that range.
    fn unmap(&mut self, id: StorageId);

    /// Releases all address spaces and cleans up all physical memory.
    fn cleanup(&mut self);

    fn are_adjacent(&self, first: &StorageHandle, second: &StorageHandle) -> bool;
}
