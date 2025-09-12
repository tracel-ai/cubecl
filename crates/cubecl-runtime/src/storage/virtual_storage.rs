use crate::server::IoError;
use crate::storage::{StorageHandle, StorageId, StorageUtilization};
use crate::storage::ComputeStorage;

enum VirtualBlockState {
    Mapped,
    Unmapped,
}

pub struct VirtualBlock {
    state: VirtualBlockState,
    pub physical_handles: Vec<StorageHandle>, //  backing physical handles
    utilization: StorageUtilization,
}

impl VirtualBlock {
    pub fn new(physical_handles: Vec<StorageHandle>, total_size: u64, offset: u64) -> Self {
        Self {
            state: VirtualBlockState::Unmapped,
            physical_handles,
            utilization: StorageUtilization {
                size: total_size,
                offset,
            },
        }
    }

    pub fn is_mapped(&self) -> bool {
        matches!(self.state, VirtualBlockState::Mapped)
    }

    pub fn is_unmapped(&self) -> bool {
        matches!(self.state, VirtualBlockState::Unmapped)
    }

    pub fn set_mapped(&mut self) {
        self.state = VirtualBlockState::Mapped;
    }

    pub fn set_unmapped(&mut self) {
        self.state = VirtualBlockState::Unmapped;
    }

    pub fn id(&self) -> StorageId {
        self.physical_handles[0].id
    }

    pub fn size(&self) -> u64 {
        self.utilization.size
    }

    pub fn offset(&self) -> u64 {
        self.utilization.offset
    }

    pub fn pages_needed(&self, page_size: u64) -> usize {
        self.size().div_ceil(page_size) as usize
    }
}

/// Trait for virtual address space management
pub trait VirtualMemoryManager: Send {
    type VirtualAddress: Send;
    type PhysicalHandle: Send;

    /// Reserves a contiguous range of virtual pages,
    /// Returns a pointer to the start of the range.
    fn reserve_range(&mut self, pages: usize) -> Result<Self::VirtualAddress, IoError>;

    /// Releases a range of virtual pages for reuse
    fn release_range(&mut self, addr: Self::VirtualAddress, pages: usize);

    /// Virtual page size
    fn page_size(&self) -> u64;

    /// Maps physical handles to a specific virtual address
    fn map_handle(
        &mut self,
        virtual_addr: Self::VirtualAddress,
        handle: Self::PhysicalHandle,
    ) -> Result<(), IoError>;

    /// Unmaps physical memory from a virtual address
    fn unmap_handle(&mut self, virtual_addr: Self::VirtualAddress) -> Result<(), IoError>;
}

/// Main storage trait that orchestrates all components
pub trait VirtualStorage: Send {
    type Resource: Send;

    /// Required alignment for allocations
    fn alignment(&self) -> usize;

    /// Gets a resource for the given storage id
    fn get(&mut self, id: StorageId) -> Self::Resource;

    /// Allocates physical memory without mapping
    /// Returns a block in Unmapped state
    fn alloc(&mut self, size: u64) -> Result<VirtualBlock, IoError>;

    /// Maps a block to virtual memory addresses
    /// Changes handle state to Mapped
    fn map(&mut self, handle: &mut VirtualBlock) -> Result<(), IoError>;

    /// Unmaps a block but keeps physical memory
    /// Changes state to Unmapped, allows reuse
    fn unmap(&mut self, id: StorageId);

    /// Marks a block for complete deallocation
    /// Unmaps if necessary and releases physical memory
    fn dealloc(&mut self, id: StorageId);

    /// Processes all pending deallocations
    fn flush(&mut self);
}



/// Lazy dealloc VirtualStorage.
/// Works exactly the same as current storages.
pub struct LazyVirtualStorageAdapter<V>
where
    V: VirtualStorage
{
    virtual_storage: V,
    deallocations: Vec<StorageId>,
}


impl<V> LazyVirtualStorageAdapter<V>
where
    V: VirtualStorage,
{
    pub fn new(virtual_storage: V) -> Self {
        Self {
            virtual_storage,
            deallocations: Vec::new(),
        }
    }
}

impl<V> ComputeStorage for LazyVirtualStorageAdapter<V>
where
    V: VirtualStorage,
{
    type Resource = V::Resource;

    fn alignment(&self) -> usize {
        self.virtual_storage.alignment()
    }

    fn get(&mut self, handle: &StorageHandle) -> V::Resource {
        self.virtual_storage.get(handle.id)
    }

    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        // 1. Allocate virtual block
        let mut virtual_block = self.virtual_storage.alloc(size)?;
        let block_id = virtual_block.id();

        // 2. Immediately map the block
        self.virtual_storage.map(&mut virtual_block)?;

        // 3. Create handle
        let handle_id = StorageId::new();
        let handle = StorageHandle::new(
            handle_id,
            StorageUtilization { offset: 0, size: virtual_block.size() }
        );


        Ok(handle)
    }

    fn dealloc(&mut self, id: StorageId) {

        self.deallocations.push(id);

    }

    fn flush(&mut self) {
        for block_id in self.deallocations.drain(..) {
            self.virtual_storage.dealloc(block_id);
        }
        self.virtual_storage.flush();
    }
}




/// Eager VirtualStorage.
/// Inmediately unmaps blocks when deallocation is requested.
// Does not completely deallocate memory, as this can be potentially be reused.
pub struct EagerVirtualStorageAdapter<V>
where
    V: VirtualStorage
{
    virtual_storage: V,
    deallocations: Vec<StorageId>,
}


impl<V> EagerVirtualStorageAdapter<V>
where
    V: VirtualStorage,
{
    pub fn new(virtual_storage: V) -> Self {
        Self {
            virtual_storage,
            deallocations: Vec::new(),
        }
    }
}

impl<V> ComputeStorage for EagerVirtualStorageAdapter<V>
where
    V: VirtualStorage,
{
    type Resource = V::Resource;

    fn alignment(&self) -> usize {
        self.virtual_storage.alignment()
    }

    fn get(&mut self, handle: &StorageHandle) -> V::Resource {
        self.virtual_storage.get(handle.id)
    }

    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        // 1. Allocate virtual block
        let mut virtual_block = self.virtual_storage.alloc(size)?;
        let block_id = virtual_block.id();

        // 2. Immediately map the block
        self.virtual_storage.map(&mut virtual_block)?;

        // 3. Create handle
        let handle_id = StorageId::new();
        let handle = StorageHandle::new(
            handle_id,
            StorageUtilization { offset: 0, size: virtual_block.size() }
        );


        Ok(handle)
    }

    fn dealloc(&mut self, id: StorageId) {
        self.virtual_storage.unmap(id); // Automatically sets the physical memory for reuse.

    }

    fn flush(&mut self) {
        self.virtual_storage.flush();
    }
}
