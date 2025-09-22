use crate::server::IoError;
use crate::storage::ComputeStorage;
use crate::storage::{StorageHandle, StorageId};

/// Virtual Storage trait.
///
/// # Notes for the reader:
///
/// When I say 'Region' I  mean a range of virtual addresses. When I say 'Block' I mean a physical memory block.
///
/// If you are familiar with the project, you might be thinking that this memory management stuff is already done at the Memory Pools' level.
/// In this case it is different, because we are working with OS based virtual memory management, which requires API calls that might not be the same for all backends. The level of flexibility that each backend provides for working with VMM can also differ.
///
///   The memory pools implement (at least at my understanding) a higher level memory management, which is software based and therefore backend-agnostic.
///
/// # Key ideas:
///
/// The idea behind the virtual storage is to abstract the management of virtual memory regions, which can be dynamically mapped or unmapped to physical blocks of a given size.
///
/// The physical blocks allocated by the virtual storage is always of the same size for  a given storage.
///
/// This allows the storage to internally reuse physical memory blocks by unampping and remapping.
///
/// The [`StorageHandles`] returned by this storage always point to virtual memory regions.
///
/// This memory regions are initially unmapped (after you call [`reserve`]), so to complete a block allocation, you must call the [`map`] method.
///
/// The storage communicates with the pools that use it using ONLY the storage handles, and provides an API for efficient and effective memory management via:
///
///     - [`split_range`]: splits an unmapped storage handle into two, at a given offset.
///     - [`expand`]: extends a virtual memory region by reserving more virtual space at the end of it.
///     - [`merge`]: attempts to merge two regions. If regions are adjacent in memory, it can just put them together. If they are not, it will release both address spaces and allocate a bigger one of the target size. Regions must be unmapped for the operation to complete sucessfully in the second case.
///     - [`defragment`]: internally defragments all unmapped regions by merging all unmapped regions into a single one.
///     - [`are_adjacent`]: utility to check if two regions are adjacent in memory.
///
/// All types implementing this trait should also implement [`ComputeStorage`].
/// You can look at `cubecl_cuda/src/compute/storage/gpu.rs` for an example.
pub trait VirtualStorage: ComputeStorage {
    /// Retrieves the minimum allocation granularity of this storage. All physical and virtual allocations should be aligned.
    fn granularity(&self) -> usize;
    /// Retrieves the size of a physical block in this virtual storage.
    fn physical_block_size(&self) -> u64;

    /// Split a range of virtual addresses into two
    fn split_range(
        &mut self,
        handle: &mut StorageHandle,
        offset: u64,
    ) -> Result<StorageHandle, IoError>;

    /// Merge two ranges of virtual addresses (either contiguous or not) into one.
    /// If the two virtual address are not contiguous they can still be merged into one as long as both are not mapped, by releasing the virtual address spaces and creating a new one.
    fn merge(
        &mut self,
        first_handle: StorageHandle,
        second_handle: StorageHandle,
    ) -> Result<StorageHandle, IoError>;

    /// Expand a virtual memory region to a target size.
    /// The difference between merge and expand is that the second one does not require the handle to be unmapped.
    fn expand(&mut self, handle: &mut StorageHandle, additional_size: u64) -> Result<(), IoError>;

    /// Reserves an address space of a given size. Padding should be automatically added to meet the granularity requirements. The parameter start_addr is the address at which the reservation should start, if applicable.
    fn reserve(&mut self, size: u64, start_addr: u64) -> Result<StorageHandle, IoError>;

    /// Releases the virtual address range associated with this handle.
    fn release(&mut self, handle: StorageHandle);

    /// Map a range of virtual addresses to physical memory.
    fn map(&mut self, handle: &mut StorageHandle) -> Result<(), IoError>;

    /// Unmap that range.
    fn unmap(&mut self, id: StorageId);

    // Check whether two handles are adjacent in memory.
    fn are_adjacent(&self, first: &StorageHandle, second: &StorageHandle) -> bool;

    /// Releases all address spaces and cleans up all physical memory.
    fn cleanup(&mut self);

    // Defragments the virtual address space, returning a new handle pointing to the defragmented region.
    fn defragment(&mut self) -> Option<StorageHandle>;
}
