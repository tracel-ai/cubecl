use crate::server::IoError;
use crate::storage::StorageUtilization;
use crate::storage::{StorageHandle, StorageId};
use crate::storage_id_type;

storage_id_type!(PhysicalStorageId);

/// Handle representing a physical memory buffer.
/// For a virtual storage, we need to be able to distinguish between physical memory and virtual memory.
/// As the virtual memory is what is going to be seen by the processes (Kernels), I created this struct while I keep the StorageHandle as a representation of a valid device pointer for kernel bindings.
#[derive(Debug, Clone)]
pub struct PhysicalStorageHandle {
    id: PhysicalStorageId,
    utilization: StorageUtilization,
    mapped: bool,
}

impl PhysicalStorageHandle {
    /// Constructor for the Physical Storage Handle
    pub fn new(id: PhysicalStorageId, utilization: StorageUtilization) -> Self {
        Self {
            id,
            utilization,
            mapped: false,
        }
    }

    /// Id of the handle
    pub fn id(&self) -> PhysicalStorageId {
        self.id
    }

    /// Whether this handle is mapped to virtual memory
    pub fn is_mapped(&self) -> bool {
        self.mapped
    }

    /// The size of the handle
    pub fn size(&self) -> u64 {
        self.utilization.size
    }

    /// Utility to set the handle mapped.
    pub fn set_mapped(&mut self, mapped: bool) {
        self.mapped = mapped
    }
}

/// Virtual Storage trait.
/// I want to make this trait optional. However, to be able to use it on the memory manager I have to restrict the type of Storage with VirtualStorage trait bounds.
/// Therefore all methods will have a default implementation.
/// By enforcing the ComputeStorage to inherit from it, all storages that implement ComputeStorage will automatically implement the default implementation of VirtualStorage.
/// Then, at runtime, the memory pools can check the method [`is_virtual_mem_enabled`] to verify is virtual memory is supported on the target backend.
pub trait VirtualStorage {
    /// Retrieves the minimum allocation granularity of this storage. All physical and virtual allocations should be aligned.
    fn granularity(&self) -> usize {
        0 // Default granularity is zero when virtual memory is not supported.
    }

    /// Check whether virtual mem is supported
    fn is_virtual_mem_enabled(&self) -> bool {
        false
    }

    /// Allocate physical memory of hthe requested size
    fn allocate(&mut self, _size: u64) -> Result<PhysicalStorageHandle, IoError> {
        Err(IoError::Unknown(
            "Virtual memory is not supported!".to_string(),
        ))
    }

    /// Releases a physical memory handle to the driver (explicit).
    fn release(&mut self, _id: PhysicalStorageId) {}

    /// Reserves an address space of a given size. Padding should be automatically added to meet the granularity requirements. The parameter `start_addr` is the id of the address space which should end at the beginning of the next reservation (if applicable).
    fn reserve(
        &mut self,
        _size: u64,
        _start_addr: Option<StorageId>,
    ) -> Result<StorageHandle, IoError> {
        Err(IoError::Unknown(
            "Virtual memory is not supported!".to_string(),
        ))
    }

    /// Releases the virtual address range associated with this handle.
    fn free(&mut self, _id: StorageId) {}

    /// Map physical memory to a range of virtual addresses
    fn map(
        &mut self,
        _id: StorageId,
        _offset: u64,
        _physical_storage: &mut PhysicalStorageHandle,
    ) -> Result<StorageHandle, IoError> {
        Err(IoError::Unknown(
            "Virtual memory is not supported!".to_string(),
        ))
    }

    /// Unmap the handles
    fn unmap(&mut self, _id: StorageId, _offset: u64, _physical: &mut PhysicalStorageHandle) {}

    /// Checks if two address spaces are contiguous in memory (the first one ends where the second one starts).
    /// This is useful to perform defragmentation.
    fn are_aligned(&self, _lhs: &StorageId, _rhs: &StorageId) -> bool {
        false
    }
}
