use crate::alloc::string::ToString;
/// This module implements a simulated virtual storage which uses heap allocations.
/// It is only intended for testing.
use crate::server::IoError;
use crate::storage::{
    ComputeStorage, PhysicalStorageHandle, PhysicalStorageId, StorageHandle, StorageId,
    StorageUtilization, VirtualStorage, bytes_cpu::BytesResource,
};
use alloc::alloc::{Layout, alloc, dealloc};
use hashbrown::HashMap;
/// Physical memory block for BytesVirtualStorage
struct BytesMemoryBlock {
    ptr: *mut u8,
    layout: Layout,
    size: u64,
    mapped: bool,
}

impl BytesMemoryBlock {
    fn new(ptr: *mut u8, layout: Layout, size: u64) -> Self {
        Self {
            ptr,
            layout,
            size,
            mapped: false,
        }
    }

    fn ptr(&self) -> *mut u8 {
        self.ptr
    }

    fn size(&self) -> u64 {
        self.size
    }

    fn set_mapped(&mut self, mapped: bool) {
        self.mapped = mapped;
    }

    fn is_mapped(&self) -> bool {
        self.mapped
    }
}

/// Virtual address space for BytesVirtualStorage
struct BytesVirtualAddressSpace {
    ptr: *mut u8,
    layout: Layout,
    size: u64,
    mappings: HashMap<u64, PhysicalStorageId>, // offset -> physical block id
}

impl BytesVirtualAddressSpace {
    fn new(ptr: *mut u8, layout: Layout, size: u64) -> Self {
        Self {
            ptr,
            layout,
            size,
            mappings: HashMap::new(),
        }
    }

    fn ptr(&self) -> *mut u8 {
        self.ptr
    }

    fn size(&self) -> u64 {
        self.size
    }

    fn add_mapping(&mut self, offset: u64, physical_id: PhysicalStorageId) {
        self.mappings.insert(offset, physical_id);
    }

    fn remove_mapping(&mut self, offset: u64) -> Option<PhysicalStorageId> {
        self.mappings.remove(&offset)
    }
}

/// Memory allocation mode for BytesVirtualStorage
#[derive(Default, Clone, Copy)]
pub enum BytesAllocationMode {
    /// Standard allocation mode (direct heap allocation)
    #[default]
    Standard = 0,
    /// Virtual memory mode (simulated with heap allocation + mapping tracking)
    Virtual = 1,
}

/// A simulated virtual memory storage using heap allocations
/// This provides the same interface as GPU virtual storage but uses CPU memory
pub struct BytesVirtualStorage {
    /// Memory alignment requirement
    mem_alignment: usize,
    /// Virtual address spaces (simulated)
    virtual_memory: HashMap<StorageId, BytesVirtualAddressSpace>,
    /// Physical memory blocks
    physical_memory: HashMap<PhysicalStorageId, BytesMemoryBlock>,
    /// Standard allocations (for ComputeStorage compatibility)
    memory: HashMap<StorageId, AllocatedBytes>,
    /// Allocation mode
    alloc_mode: BytesAllocationMode,
}

/// Standard allocated bytes (for ComputeStorage mode)
struct AllocatedBytes {
    ptr: *mut u8,
    layout: Layout,
}

unsafe impl Send for BytesVirtualStorage {}

impl BytesVirtualStorage {
    /// Constructor for the BytesVirtualStorage
    pub fn new(mem_alignment: usize, alloc_mode: BytesAllocationMode) -> Self {
        Self {
            mem_alignment,
            virtual_memory: HashMap::new(),
            physical_memory: HashMap::new(),
            memory: HashMap::new(),
            alloc_mode,
        }
    }

    /// Allocate a physical memory block
    fn allocate_physical_block(&mut self, size: u64) -> Result<BytesMemoryBlock, IoError> {
        assert_eq!(
            size % self.granularity() as u64,
            0,
            "For virtual memory allocations, size must be aligned to granularity"
        );

        unsafe {
            let layout = Layout::array::<u8>(size as usize)
                .map_err(|_| IoError::BufferTooBig(size as usize))?;

            let ptr = alloc(layout);
            if ptr.is_null() {
                return Err(IoError::BufferTooBig(size as usize));
            }

            // Initialize memory to zero for consistency
            core::ptr::write_bytes(ptr, 0, size as usize);

            Ok(BytesMemoryBlock::new(ptr, layout, size))
        }
    }

    /// Simulate mapping by copying data from physical block to virtual address space
    fn simulate_map(
        &mut self,
        virtual_addr: *mut u8,
        offset: u64,
        physical_ptr: *mut u8,
        size: u64,
    ) {
        unsafe {
            let dest = virtual_addr.add(offset as usize);
            core::ptr::copy_nonoverlapping(physical_ptr, dest, size as usize);
        }
    }

    /// Simulate unmapping by copying data back from virtual address space to physical block
    fn simulate_unmap(
        &mut self,
        virtual_addr: *mut u8,
        offset: u64,
        physical_ptr: *mut u8,
        size: u64,
    ) {
        unsafe {
            let src = virtual_addr.add(offset as usize);
            core::ptr::copy_nonoverlapping(src, physical_ptr, size as usize);
        }
    }
}

impl VirtualStorage for BytesVirtualStorage {
    fn granularity(&self) -> usize {
        self.mem_alignment
    }

    fn is_virtual_mem_enabled(&self) -> bool {
        matches!(self.alloc_mode, BytesAllocationMode::Virtual)
    }

    fn allocate(&mut self, size: u64) -> Result<PhysicalStorageHandle, IoError> {
        if !self.is_virtual_mem_enabled() {
            return Err(IoError::Unknown("Virtual memory is disabled!".to_string()));
        }

        let total_size = size.next_multiple_of(self.mem_alignment as u64);
        let block = self.allocate_physical_block(total_size)?;

        let id = PhysicalStorageId::new();
        let phys = PhysicalStorageHandle::new(id, StorageUtilization { size, offset: 0 });
        self.physical_memory.insert(id, block);

        Ok(phys)
    }

    fn are_aligned(&self, lhs: &StorageId, rhs: &StorageId) -> bool {
        if let (Some(a), Some(b)) = (self.virtual_memory.get(lhs), self.virtual_memory.get(rhs)) {
            // Check if the first address space ends where the second begins
            let a_end = a.ptr() as u64 + a.size();

            let b_start = b.ptr() as u64;

            return a_end == b_start;
        }
        false
    }

    // Note: Since we're using heap allocation, alignment is not guaranteed
    // but we can still test the function works
    fn reserve(
        &mut self,
        size: u64,
        _start_addr: Option<StorageId>,
    ) -> Result<StorageHandle, IoError> {
        if !self.is_virtual_mem_enabled() {
            return Err(IoError::Unknown("Virtual memory is disabled!".to_string()));
        }

        let aligned_size = size.next_multiple_of(self.mem_alignment as u64);

        unsafe {
            let layout = Layout::array::<u8>(aligned_size as usize)
                .map_err(|_| IoError::BufferTooBig(aligned_size as usize))?;

            let ptr = alloc(layout);
            if ptr.is_null() {
                return Err(IoError::BufferTooBig(aligned_size as usize));
            }

            // Initialize virtual address space to zero
            core::ptr::write_bytes(ptr, 0, aligned_size as usize);

            let id = StorageId::new();
            let addr_space = BytesVirtualAddressSpace::new(ptr, layout, aligned_size);

            self.virtual_memory.insert(id, addr_space);

            let handle = StorageHandle::new(id, StorageUtilization { size, offset: 0 });
            Ok(handle)
        }
    }

    fn free(&mut self, id: StorageId) {
        if let Some(addr_space) = self.virtual_memory.remove(&id) {
            unsafe {
                dealloc(addr_space.ptr, addr_space.layout);
            }
        }
    }

    fn release(&mut self, id: PhysicalStorageId) {
        if let Some(block) = self.physical_memory.remove(&id) {
            assert!(!block.is_mapped(), "Cannot release a mapped handle!");
            unsafe {
                dealloc(block.ptr, block.layout);
            }
        }
    }

    fn map(
        &mut self,
        id: StorageId,
        offset: u64,
        physical: &mut PhysicalStorageHandle,
    ) -> Result<StorageHandle, IoError> {
        if !self.is_virtual_mem_enabled() {
            return Err(IoError::Unknown("Virtual memory is disabled!".to_string()));
        }

        let aligned_offset = offset.next_multiple_of(self.mem_alignment as u64);

        // Get pointers and sizes first to avoid borrowing conflicts
        let (virtual_ptr, virtual_size) = {
            let space = self.virtual_memory.get(&id).ok_or(IoError::InvalidHandle)?;
            (space.ptr(), space.size())
        };

        let (physical_ptr, ph_size) = {
            let ph = self
                .physical_memory
                .get(&physical.id())
                .ok_or(IoError::InvalidHandle)?;
            (ph.ptr(), ph.size())
        };

        if (aligned_offset + ph_size) > virtual_size {
            return Err(IoError::InvalidHandle);
        }

        // Simulate mapping by copying data
        self.simulate_map(virtual_ptr, aligned_offset, physical_ptr, ph_size);

        // Track the mapping
        let space = self
            .virtual_memory
            .get_mut(&id)
            .ok_or(IoError::InvalidHandle)?;
        space.add_mapping(aligned_offset, physical.id());

        physical.set_mapped(true);

        let ph_mut = self
            .physical_memory
            .get_mut(&physical.id())
            .expect("Physical storage not found");
        ph_mut.set_mapped(true);

        let handle = StorageHandle::new(
            id,
            StorageUtilization {
                offset,
                size: ph_size,
            },
        );
        Ok(handle)
    }

    fn unmap(&mut self, id: StorageId, offset: u64, physical: &mut PhysicalStorageHandle) {
        let aligned_offset = offset.next_multiple_of(self.mem_alignment as u64);

        // Get pointers and size first to avoid borrowing conflicts
        let (virtual_ptr, physical_ptr, aligned_size) = {
            let ph = match self.physical_memory.get(&physical.id()) {
                Some(ph) => ph,
                None => return,
            };
            let aligned_size = ph.size();

            let mapping = match self.virtual_memory.get(&id) {
                Some(mapping) => mapping,
                None => return,
            };

            (mapping.ptr(), ph.ptr(), aligned_size)
        };

        // Simulate unmapping by copying data back
        self.simulate_unmap(virtual_ptr, aligned_offset, physical_ptr, aligned_size);

        // Remove mapping tracking and update states
        if let Some(mapping) = self.virtual_memory.get_mut(&id) {
            mapping.remove_mapping(aligned_offset);
        }

        if let Some(ph) = self.physical_memory.get_mut(&physical.id()) {
            physical.set_mapped(false);
            ph.set_mapped(false);
        }
    }
}

impl ComputeStorage for BytesVirtualStorage {
    type Resource = BytesResource;

    fn alignment(&self) -> usize {
        self.mem_alignment
    }

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let ptr = match self.alloc_mode {
            BytesAllocationMode::Standard => {
                self.memory
                    .get(&handle.id)
                    .expect("Storage handle not found")
                    .ptr
            }
            BytesAllocationMode::Virtual => self
                .virtual_memory
                .get(&handle.id)
                .expect("Storage handle not found")
                .ptr(),
        };

        BytesResource {
            ptr,
            utilization: handle.utilization.clone(),
        }
    }

    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        match self.alloc_mode {
            BytesAllocationMode::Standard => {
                let id = StorageId::new();
                let handle = StorageHandle::new(id, StorageUtilization { offset: 0, size });

                unsafe {
                    let layout = Layout::array::<u8>(size as usize)
                        .map_err(|_| IoError::BufferTooBig(size as usize))?;
                    let ptr = alloc(layout);
                    if ptr.is_null() {
                        return Err(IoError::BufferTooBig(size as usize));
                    }
                    let memory = AllocatedBytes { ptr, layout };
                    self.memory.insert(id, memory);
                }

                Ok(handle)
            }
            BytesAllocationMode::Virtual => {
                // For virtual mode, allocate physical + reserve virtual + map
                let aligned_size = size.next_multiple_of(self.mem_alignment as u64);

                // Allocate physical memory
                let mut physical_handle = self.allocate(aligned_size)?;

                // Reserve virtual address space
                let virtual_handle = self.reserve(aligned_size, None)?;

                // Map physical to virtual
                let mapped_handle = self.map(virtual_handle.id, 0, &mut physical_handle)?;

                Ok(StorageHandle::new(
                    mapped_handle.id,
                    StorageUtilization { offset: 0, size },
                ))
            }
        }
    }

    fn dealloc(&mut self, id: StorageId) {
        match self.alloc_mode {
            BytesAllocationMode::Standard => {
                if let Some(memory) = self.memory.remove(&id) {
                    unsafe {
                        dealloc(memory.ptr, memory.layout);
                    }
                }
            }
            BytesAllocationMode::Virtual => {
                // For virtual mode, we need to clean up the virtual address space
                self.free(id);
            }
        }
    }

    fn flush(&mut self) {
        // No-op for bytes storage as we don't have async deallocations
    }
}

impl Default for BytesVirtualStorage {
    fn default() -> Self {
        Self::new(4, BytesAllocationMode::default())
    }
}

impl Drop for BytesVirtualStorage {
    fn drop(&mut self) {
        self.flush();

        // Clean up all virtual address spaces
        for (_, addr_space) in self.virtual_memory.drain() {
            unsafe {
                dealloc(addr_space.ptr, addr_space.layout);
            }
        }

        // Clean up all physical memory blocks
        for (_, block) in self.physical_memory.drain() {
            unsafe {
                dealloc(block.ptr, block.layout);
            }
        }

        // Clean up standard allocations
        for (_, memory) in self.memory.drain() {
            unsafe {
                dealloc(memory.ptr, memory.layout);
            }
        }
    }
}

impl core::fmt::Debug for BytesVirtualStorage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BytesVirtualStorage")
            .field("mem_alignment", &self.mem_alignment)
            .field("alloc_mode", &(self.alloc_mode as u32))
            .field("virtual_memory_count", &self.virtual_memory.len())
            .field("physical_memory_count", &self.physical_memory.len())
            .field("memory_count", &self.memory.len())
            .finish()
    }
}
