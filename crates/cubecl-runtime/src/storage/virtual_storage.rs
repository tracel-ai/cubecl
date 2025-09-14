
//! Virtual Memory Management System
//!
//! This module implements a virtual memory abstraction layer that decouples virtual address spaces
//! from physical memory allocations, enabling advanced memory management strategies such as:
//! - Memory reuse across different allocation sizes
//! - Lazy physical memory allocation
//! - Fine-grained control over memory mapping and unmapping
//! - Efficient handling of fragmented physical memory
use crate::server::IoError;
use crate::{storage_id_type, storage::{StorageHandle, StorageId, StorageUtilization}};



// Unique identifier for virtual address spaces.
//
// Virtual address spaces represent ranges of addresses visible to the program,
// which may or may not be backed by physical memory at any given time.
storage_id_type!(VirtualSpaceId);

// Unique identifier for physical memory allocations.
//
// Physical storage represents actual hardware memory that has been allocated
// from the device but may not yet be mapped to any virtual address space.
storage_id_type!(PhysicalStorageId);

/// Handle representing a reserved virtual address space.
///
/// Virtual address spaces are ranges of addresses that appear contiguous to the program
/// but are not necessarily backed by contiguous physical memory. This handle tracks
/// reservation state and size information for a virtual address range.
///
/// # Lifecycle
///
/// 1. **Reserved**: Created via `try_reserve()` - address space is reserved but unmapped
/// 2. **Mapped**: Associated with physical memory via `try_map()`
/// 3. **Unmapped**: Physical memory removed but virtual space remains reserved
/// 4. **Released**: Virtual address space returned to the system via `try_release()`
#[derive(Debug, Clone)]
pub struct VirtualAddressSpaceHandle {
    /// Unique identifier for this virtual address space
    id: VirtualSpaceId,
    /// Size and offset information for the virtual address range
    utilization: StorageUtilization,

}

impl VirtualAddressSpaceHandle {
    /// Creates a new virtual address space handle.
    ///
    /// # Arguments
    /// * `id` - Unique identifier for this virtual space
    /// * `utilization` - Size and offset information
    pub fn new(id: VirtualSpaceId, utilization: StorageUtilization) -> Self {
        Self {
            id,
            utilization,
        }
    }

    /// Returns the unique identifier for this virtual address space.
    pub fn id(&self) -> VirtualSpaceId {
        self.id
    }

    /// Returns the total size of this virtual address space in bytes.
    pub fn size(&self) -> u64 {
        self.utilization.size
    }

    /// Returns the offset within the virtual address space.
    pub fn offset(&self) -> u64 {
        self.utilization.offset
    }

}

/// Represents the relationship between virtual address spaces and physical memory.
///
/// This structure maintains the bidirectional mapping between virtual address spaces
/// and physical memory allocations. It serves as the core data structure for tracking
/// active memory mappings in the virtual memory system.
pub struct VirtualMapping {
    /// Physical memory allocation backing this mapping
    pub physical_id: PhysicalStorageId,
    /// Virtual address space where the physical memory is mapped
    pub virtual_id: VirtualSpaceId,
}

impl VirtualMapping {
    /// Creates a new virtual-to-physical mapping.
    ///
    /// # Arguments
    /// * `physical_id` - The physical memory allocation
    /// * `virtual_id` - The virtual address space where it's mapped
    pub fn new(physical_id: PhysicalStorageId, virtual_id: VirtualSpaceId) -> Self {
        Self {
            physical_id,
            virtual_id,
        }
    }


}

/// Handle representing a physical memory allocation.
///
/// Physical memory handles track actual hardware memory allocations that exist
/// independently of virtual address mappings. A single physical allocation can
/// potentially be mapped to multiple virtual address spaces (though this is
/// implementation-dependent).
///
/// # Memory Lifecycle
///
/// 1. **Allocated**: Created via `alloc()` - physical memory is reserved from hardware
/// 2. **Mapped**: Associated with virtual address space via `try_map()`
/// 3. **Unmapped**: Removed from virtual address space but physical memory remains
/// 4. **Freed**: Physical memory returned to hardware via `try_free()`
#[derive(Debug, Clone)]
pub struct PhysicalStorageHandle {
    /// Unique identifier for this physical memory allocation
    id: PhysicalStorageId,
    /// Size and offset information for the physical memory block
    utilization: StorageUtilization,

}

impl PhysicalStorageHandle {
    /// Creates a new physical storage handle.
    ///
    /// # Arguments
    /// * `id` - Unique identifier for this physical allocation
    /// * `utilization` - Size and offset information
    pub fn new(id: PhysicalStorageId, utilization: StorageUtilization) -> Self {
        Self {
            id,
            utilization,

        }
    }

    /// Returns the unique identifier for this physical memory allocation.
    pub fn id(&self) -> PhysicalStorageId {
        self.id
    }

    /// Returns the size of this physical memory allocation in bytes.
    pub fn size(&self) -> u64 {
        self.utilization.size
    }

    /// Returns the offset within the physical memory block.
    pub fn offset(&self) -> u64 {
        self.utilization.offset
    }


}

/// Virtual Memory Storage System
///
/// This trait defines a virtual memory management interface that provides fine-grained
/// control over memory allocation, mapping, and deallocation. It extends the traditional
/// allocator model by separating concerns between:
///
/// ## Core Concepts
///
/// ### Virtual Address Spaces
/// Ranges of addresses (e.g., 0x0000_0000..0xFFFF_FFFF) that appear contiguous to programs.
/// These addresses are what pointers see and what the program uses for memory access.
///
/// ### Physical Memory
/// Actual hardware memory frames where data is stored. Physical frames can be scattered
/// throughout device memory and don't need to be contiguous.
///
/// ### Memory Mapping
/// The translation mechanism that connects virtual addresses to physical memory locations.
/// This is maintained by the storage implementation and is transparent to the program.
///
/// ## Architecture Integration
///
/// This trait is designed to integrate with existing memory management systems:
///
/// 1. **Memory Pools**: Higher-level allocators that implement reuse strategies, block
///    merging, and splitting. They use this interface to request and manage memory.
///
/// 2. **Storage Handles**: The primary communication mechanism between memory pools and
///    storage. Handles represent mapped memory that is ready for program use.
///
/// 3. **Compute Storage**: Hardware-specific implementations that interface with vendor
///    APIs (CUDA, Vulkan, etc.) to perform actual memory operations.
///
/// ## Memory Lifecycle
///
/// The typical workflow involves these phases:
///
/// 1. **Reserve**: Allocate virtual address space (`try_reserve`)
/// 2. **Allocate**: Create physical memory allocation (`alloc`)
/// 3. **Map**: Connect virtual and physical memory (`try_map`)
/// 4. **Use**: Access memory through returned `StorageHandle`
/// 5. **Unmap**: Disconnect virtual-physical relationship (`try_unmap`)
/// 6. **Free**: Return physical memory to hardware (`try_free`)
/// 7. **Release**: Return virtual address space to system (`try_release`)
///
/// ## Benefits
///
/// - **Memory Reuse**: Physical allocations can be reused across different virtual spaces
/// - **Fragmentation Management**: Virtual memory appears contiguous regardless of physical layout
/// - **Lazy Allocation**: Physical memory can be allocated only when needed
/// - **Fine Control**: Memory pools can implement sophisticated allocation strategies
/// - **Hardware Abstraction**: Vendor-specific details are isolated in implementations
pub trait VirtualStorage: Send {
    /// The resource type returned when accessing mapped memory.
    /// This should provide the same semantics as traditional compute storage resources.
    type Resource: Send;

    /// Returns the required memory alignment in bytes.
    ///
    /// All allocations will be aligned to this boundary, which typically corresponds
    /// to the hardware page size or other device-specific requirements.
    fn alignment(&self) -> usize;

    /// Reserves a contiguous range of virtual addresses.
    ///
    /// This operation only reserves address space from the operating system or device
    /// driver. No physical memory is allocated, and the addresses cannot be used until
    /// they are mapped to physical memory.
    ///
    /// # Arguments
    /// * `total_size` - Size in bytes of virtual address space to reserve
    ///
    /// # Returns
    /// * `Ok(VirtualAddressSpaceHandle)` - Handle representing the reserved address space
    /// * `Err(IoError::BufferTooBig)` - If the requested size cannot be reserved
    fn try_reserve(&mut self, total_size: usize) -> Result<VirtualAddressSpaceHandle, IoError>;

    /// Releases a virtual address space back to the system.
    ///
    /// This operation frees the virtual address range, making it available for future
    /// reservations. Any active mappings using this virtual space should be unmapped
    /// before calling this method.
    ///
    /// # Arguments
    /// * `id` - Identifier of the virtual space to release
    ///
    /// # Returns
    /// * `Ok(())` - Virtual space successfully released
    /// * `Err(IoError::InvalidHandle)` - If the virtual space ID is invalid or still mapped
    fn try_release(&mut self, id: VirtualSpaceId) -> Result<(), IoError>;

    /// Retrieves a resource handle for mapped memory.
    ///
    /// This method converts a storage handle (which represents mapped virtual memory)
    /// into a resource that can be used for actual memory operations. The resource
    /// provides the interface needed by compute operations.
    ///
    /// # Arguments
    /// * `handle` - Storage handle representing mapped memory
    ///
    /// # Returns
    /// A resource object that can be used for memory access
    ///
    /// # Panics
    /// May panic if the handle refers to unmapped or deallocated memory
    fn get(&mut self, handle: &StorageHandle) -> Self::Resource;

    /// Allocates physical memory from the hardware.
    ///
    /// This operation creates a physical memory allocation but does not map it to any
    /// virtual address space. The allocation is hardware-specific and may involve
    /// communication with device drivers.
    ///
    /// # Arguments
    /// * `size` - Size in bytes to allocate (will be aligned automatically)
    ///
    /// # Returns
    /// * `Ok(PhysicalStorageHandle)` - Handle representing the physical allocation
    /// * `Err(IoError::BufferTooBig)` - If the allocation fails due to insufficient memory
    fn alloc(&mut self, size: u64) -> Result<PhysicalStorageHandle, IoError>;

    /// Maps physical memory to a virtual address space.
    ///
    /// This operation establishes the connection between virtual addresses and physical
    /// memory, making the memory accessible through the virtual address space. After
    /// successful mapping, the returned storage handle can be used for memory access.
    ///
    /// # Arguments
    /// * `virtual_addr` - Virtual address space where memory should be mapped
    /// * `handle` - Physical memory allocation to map
    ///
    /// # Returns
    /// * `Ok(StorageHandle)` - Handle for accessing the mapped memory
    /// * `Err(IoError::InvalidHandle)` - If mapping fails or handles are invalid
    fn try_map(
        &mut self,
        virtual_addr: VirtualAddressSpaceHandle,
        handle: PhysicalStorageHandle,
    ) -> Result<StorageHandle, IoError>;

    /// Unmaps memory from a virtual address space.
    ///
    /// This operation breaks the connection between virtual addresses and physical memory
    /// without affecting the underlying allocations. The virtual address space remains
    /// reserved and the physical memory remains allocated for potential reuse.
    ///
    /// # Arguments
    /// * `id` - Storage ID of the mapping to remove
    ///
    /// # Returns
    /// * `Ok(())` - Memory successfully unmapped
    /// * `Err(IoError::InvalidHandle)` - If the storage ID is invalid or not mapped
    fn try_unmap(&mut self, id: StorageId) -> Result<(), IoError>;

    /// Frees physical memory back to the hardware.
    ///
    /// This operation permanently releases physical memory back to the device or driver.
    /// The memory becomes unavailable for use and any existing mappings should be
    /// removed before calling this method.
    ///
    /// # Arguments
    /// * `id` - Physical storage ID to free
    ///
    /// # Returns
    /// * `Ok(())` - Memory successfully freed
    /// * `Err(IoError::InvalidHandle)` - If the physical storage ID is invalid
    fn try_free(&mut self, id: PhysicalStorageId) -> Result<(), IoError>;

    /// Marks a mapped memory block for deallocation.
    ///
    /// This method provides a high-level interface for memory pools to indicate that
    /// memory is no longer needed. The implementation may choose to immediately unmap
    /// and free the memory, or defer the operation until `flush()` is called.
    ///
    /// This design allows for:
    /// - Batched deallocation to reduce API call overhead
    /// - Lazy cleanup strategies
    /// - Memory reuse opportunities
    ///
    /// # Arguments
    /// * `id` - Storage ID of the mapped memory to deallocate
    fn dealloc(&mut self, id: StorageId);

    /// Processes all pending memory operations.
    ///
    /// This method ensures that all deferred operations (deallocations, unmapping, etc.)
    /// are completed and resources are returned to the system. It should be called
    /// periodically to prevent excessive memory usage from accumulated pending operations.
    ///
    /// Implementations should:
    /// - Process all pending deallocations
    /// - Free unused physical memory
    /// - Release unused virtual address spaces
    /// - Perform any necessary cleanup operations
    fn flush(&mut self);
}
