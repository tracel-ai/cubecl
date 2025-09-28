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
    /// Creates a new physical memory block in the heap of the specified size and with the specified Layout
    fn new(ptr: *mut u8, layout: Layout, size: u64) -> Self {
        Self {
            ptr,
            layout,
            size,
            mapped: false,
        }
    }

    /// Returns the pointer to this block in the heap.
    fn ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Returns the size of the block
    fn size(&self) -> u64 {
        self.size
    }

    /// Sets the block to mapped state
    fn set_mapped(&mut self, mapped: bool) {
        self.mapped = mapped;
    }

    /// Checks whether the block is mapped
    fn is_mapped(&self) -> bool {
        self.mapped
    }
}

/// Entry in the PageTable to address translation (Virtual to physical.)
#[derive(Debug, Clone, Copy)]
struct PageTableEntry {
    physical_id: PhysicalStorageId,
    physical_offset: u64,
    size: u64,
}

/// Virtual address space for BytesVirtualStorage
struct BytesVirtualAddressSpace {
    base_addr: u64, // Base virtual address (it is simulated)
    size: u64,
    // The page table maps virtual addresses to physical memory.
    page_table: HashMap<u64, PageTableEntry>,
}

impl BytesVirtualAddressSpace {
    /// Constructor for this Virtual Address Space
    fn new(base_addr: u64, size: u64) -> Self {
        Self {
            base_addr,
            size,
            page_table: HashMap::new(),
        }
    }

    /// Getter for the base address
    fn base_addr(&self) -> u64 {
        self.base_addr
    }

    /// Getter for the size.
    fn size(&self) -> u64 {
        self.size
    }

    /// Add a mapping at a specific offset
    fn add_mapping(
        &mut self,
        virtual_offset: u64,
        physical_id: PhysicalStorageId,
        physical_offset: u64,
        size: u64,
    ) {
        let entry = PageTableEntry {
            physical_id,
            physical_offset,
            size,
        };
        self.page_table.insert(virtual_offset, entry);
    }

    /// Remove a mapping from a specific offset.
    fn remove_mapping(&mut self, virtual_offset: u64) -> Option<PageTableEntry> {
        self.page_table.remove(&virtual_offset)
    }

    /// Translate virtual address to physical pointer
    fn translate(
        &self,
        virtual_offset: u64,
        physical_memory: &HashMap<PhysicalStorageId, BytesMemoryBlock>,
    ) -> Option<*mut u8> {
        // Lookup for this address in the page table.
        for (&page_offset, entry) in &self.page_table {
            if virtual_offset >= page_offset
                && virtual_offset < (page_offset + entry.size)
                && let Some(physical_block) = physical_memory.get(&entry.physical_id)
            {
                let offset_in_page = virtual_offset - page_offset;
                let physical_offset = entry.physical_offset + offset_in_page;
                unsafe {
                    return Some(physical_block.ptr().add(physical_offset as usize));
                }
            }
        }
        None
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
    /// Counter for generating unique virtual base addresses
    next_virtual_addr: u64,
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
            next_virtual_addr: 0x1000_0000, // Start at a simulated base address
        }
    }

    /// Generate a unique virtual base address
    fn allocate_virtual_address(&mut self, size: u64) -> u64 {
        let addr = self.next_virtual_addr;
        self.next_virtual_addr += size.next_multiple_of(4096); // Hard coded value.
        addr
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

    /// Translate virtual address to physical pointer for a given virtual storage
    fn translate_address(&self, storage_id: StorageId, virtual_offset: u64) -> Option<*mut u8> {
        self.virtual_memory
            .get(&storage_id)?
            .translate(virtual_offset, &self.physical_memory)
    }
}

/// Virtual storage implementation.
impl VirtualStorage for BytesVirtualStorage {
    /// Get the current allocation granularity
    fn granularity(&self) -> usize {
        self.mem_alignment
    }

    /// Check whether virtual memory is enabled.
    fn is_virtual_mem_enabled(&self) -> bool {
        matches!(self.alloc_mode, BytesAllocationMode::Virtual)
    }

    /// Allocate physical memory
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

    /// Checks whether to virtual addresses are aligned.
    fn are_aligned(&self, lhs: &StorageId, rhs: &StorageId) -> bool {
        if let (Some(a), Some(b)) = (self.virtual_memory.get(lhs), self.virtual_memory.get(rhs)) {
            // Check if the first address space ends where the second begins
            let a_end = a.base_addr() + a.size();
            let b_start = b.base_addr();
            return a_end == b_start;
        }
        false
    }

    /// Reserves a simulated virtual address space
    fn reserve(
        &mut self,
        size: u64,
        start_addr: Option<StorageId>,
    ) -> Result<StorageHandle, IoError> {
        if !self.is_virtual_mem_enabled() {
            return Err(IoError::Unknown("Virtual memory is disabled!".to_string()));
        }

        let aligned_size = size.next_multiple_of(self.mem_alignment as u64);

        // Here I am trying to do something similar to what I guess the CUDA driver does when called with reserve.
        // Check if the requested address is available. If not, just allocate at a new one.
        let base_addr = self.allocate_virtual_address(aligned_size);

        let provided_addr = if let Some(start) = start_addr {
            if let Some(space) = self.virtual_memory.get(&start) {
                space.base_addr + space.size()
            } else {
                0
            }
        } else {
            0
        };

        if provided_addr != base_addr {
            eprintln!(
                "Start address provided is not available. Will allocate at: {}",
                base_addr
            );
        };

        let id = StorageId::new();
        let addr_space = BytesVirtualAddressSpace::new(base_addr, aligned_size);

        self.virtual_memory.insert(id, addr_space);

        let handle = StorageHandle::new(id, StorageUtilization { size, offset: 0 });
        Ok(handle)
    }

    /// Removes a virtual address space from the table.
    fn free(&mut self, id: StorageId) {
        // Remove from page table.
        self.virtual_memory.remove(&id);
    }

    /// Releases physical memory back to the heap.
    fn release(&mut self, id: PhysicalStorageId) {
        if let Some(block) = self.physical_memory.remove(&id) {
            assert!(!block.is_mapped(), "Cannot release a mapped handle!");
            unsafe {
                dealloc(block.ptr, block.layout);
            }
        }
    }

    /// Simulates a mapping between a virtual address space and a physical block by inserting an entry in the page table.
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

        // Verify the physical block exists.
        let ph_size = {
            let ph = self
                .physical_memory
                .get(&physical.id())
                .ok_or(IoError::InvalidHandle)?;
            ph.size()
        };

        // Verify that the address space exists and has enough size
        let virtual_size = {
            let space = self.virtual_memory.get(&id).ok_or(IoError::InvalidHandle)?;
            space.size()
        };

        if (aligned_offset + ph_size) > virtual_size {
            return Err(IoError::InvalidHandle);
        }

        // Add the entry to the page table
        let space = self
            .virtual_memory
            .get_mut(&id)
            .ok_or(IoError::InvalidHandle)?;

        space.add_mapping(aligned_offset, physical.id(), 0, ph_size);

        // Mark as mapped
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

    /// Removes an entry from the page table, simulating how OS performs unmap.
    fn unmap(&mut self, id: StorageId, offset: u64, physical: &mut PhysicalStorageHandle) {
        let aligned_offset = offset.next_multiple_of(self.mem_alignment as u64);

        // Remove an entry from the page table.
        if let Some(mapping) = self.virtual_memory.get_mut(&id) {
            mapping.remove_mapping(aligned_offset);
        }

        // Update states.
        if let Some(ph) = self.physical_memory.get_mut(&physical.id()) {
            physical.set_mapped(false);
            ph.set_mapped(false);
        }
    }
}

/// Compute storage implementation for BytesVirtualStorage mostly resembles BytesCpu behaviour.
impl ComputeStorage for BytesVirtualStorage {
    type Resource = BytesResource;

    /// Gets the mmeory alignment requirement
    fn alignment(&self) -> usize {
        self.mem_alignment
    }

    /// Gets a handle from regular or virtual memory, depending on storage mode.
    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let ptr = match self.alloc_mode {
            BytesAllocationMode::Standard => {
                self.memory
                    .get(&handle.id)
                    .expect("Storage handle not found")
                    .ptr
            }
            BytesAllocationMode::Virtual => {
                // On virtual mode we need to perform address translation.
                self.translate_address(handle.id, handle.utilization.offset)
                    .expect("Virtual address translation failed")
            }
        };

        BytesResource {
            ptr,
            utilization: handle.utilization.clone(),
        }
    }

    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
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

    fn dealloc(&mut self, id: StorageId) {
        if let Some(memory) = self.memory.remove(&id) {
            unsafe {
                dealloc(memory.ptr, memory.layout);
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

        // Clean up all virtual address spaces (no hay memoria física que liberar)
        self.virtual_memory.clear();

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
            .field(
                "next_virtual_addr",
                &format!("0x{:x}", self.next_virtual_addr),
            )
            .finish()
    }
}
