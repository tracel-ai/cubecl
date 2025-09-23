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

    /// Allocate physical memory of hthe requested size
    fn allocate(&mut self, size: u64) -> Result<(), IoError>;

    /// Reserves an address space of a given size. Padding should be automatically added to meet the granularity requirements. The parameter start_addr is the address at which the reservation should start, if applicable.
    fn reserve(&mut self, size: u64, start_addr: u64) -> Result<StorageHandle, IoError>;

    /// Releases the virtual address range associated with this handle.
    fn release(&mut self, id: StorageId);

    /// Map physical handles to a range of virtual addresses
    fn map(&mut self, id: StorageId, offset: u64, size: u64) -> Result<StorageHandle, IoError>;

    /// Unmap the handles
    fn unmap(&mut self, id: StorageId, offset: u64, size: u64);

    // Check whether two address ranges are adjacent in memory.
    fn are_adjacent(&self, first: &StorageHandle, second: &StorageHandle) -> bool;

    /// Releases all address spaces and cleans up all physical memory.
    fn cleanup(&mut self);
}
