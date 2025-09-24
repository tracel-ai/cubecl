use crate::server::IoError;
use crate::storage::ComputeStorage;
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
pub trait VirtualStorage: ComputeStorage {
    /// Retrieves the minimum allocation granularity of this storage. All physical and virtual allocations should be aligned.
    fn granularity(&self) -> usize;

    /// Allocate physical memory of hthe requested size
    fn allocate(&mut self, size: u64) -> Result<PhysicalStorageHandle, IoError>;

    /// Releases a physical memory handle to the driver (explicit).
    fn release(&mut self, id: PhysicalStorageId);

    /// Reserves an address space of a given size. Padding should be automatically added to meet the granularity requirements. The parameter start_addr is the address at which the reservation should start, if applicable.
    fn reserve(&mut self, size: u64, start_addr: u64) -> Result<StorageHandle, IoError>;

    /// Releases the virtual address range associated with this handle.
    fn free(&mut self, id: StorageId);

    /// Map physical memory to a range of virtual addresses
    fn map(
        &mut self,
        id: StorageId,
        offset: u64,
        physical_storage: &mut PhysicalStorageHandle,
    ) -> Result<StorageHandle, IoError>;

    /// Unmap the handles
    fn unmap(&mut self, id: StorageId, offset: u64, physical: &mut PhysicalStorageHandle);

}
