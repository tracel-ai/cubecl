use super::uninit_vec;
use cubecl_core::server::IoError;
use cubecl_runtime::storage::{
    ComputeStorage, StorageHandle, StorageId, StorageUtilization,
};
use cubecl_runtime::storage::{VirtualBlock, VirtualStorage, VirtualMemoryManager};
use cudarc::driver::DriverError;

use cudarc::driver::sys::{
    CUdeviceptr, CUmemAccess_flags_enum::CU_MEM_ACCESS_FLAGS_PROT_READWRITE, CUmemAccessDesc,
    CUmemAllocationHandleType_enum, CUmemAllocationProp,
    CUmemAllocationType_enum::CU_MEM_ALLOCATION_TYPE_PINNED, CUmemGenericAllocationHandle,
    CUmemLocation, CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_DEVICE, CUstream, cuMemAddressFree,
    cuMemAddressReserve, cuMemCreate, cuMemGetInfo_v2, cuMemMap, cuMemRelease, cuMemSetAccess,
    cuMemUnmap, cudaError_enum::CUDA_SUCCESS,
};
use cudarc::driver::sys::cuMemGetAllocationGranularity;
use std::collections::{HashMap, VecDeque};

/// Buffer storage for cuda.
pub struct CudaStorage {
    memory: HashMap<StorageId, cudarc::driver::sys::CUdeviceptr>,
    deallocations: Vec<StorageId>,
    stream: cudarc::driver::sys::CUstream,
    ptr_bindings: PtrBindings,
    mem_alignment: usize,
}

struct PtrBindings {
    slots: Vec<cudarc::driver::sys::CUdeviceptr>,
    cursor: usize,
}

impl PtrBindings {
    fn new(max_bindings: Option<usize>) -> Self {
        let num_slots = match max_bindings {
            Some(nbind) => nbind,
            None => crate::device::CUDA_MAX_BINDINGS as usize,
        };

        Self {
            slots: uninit_vec(num_slots),
            cursor: 0,
        }
    }

    fn register(&mut self, ptr: u64) -> &u64 {
        self.slots[self.cursor] = ptr;
        let ptr = self.slots.get(self.cursor).unwrap();

        self.cursor += 1;

        // Reset the cursor.
        if self.cursor >= self.slots.len() {
            self.cursor = 0;
        }

        ptr
    }
}

unsafe impl Send for CudaStorage {}

impl core::fmt::Debug for CudaStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(format!("CudaStorage {{ device: {:?} }}", self.stream).as_str())
    }
}

/// Keeps actual CUDA buffer references in a hashmap with ids as keys.
impl CudaStorage {
    /// Create a new storage on the given [device](cudarc::driver::sys::CUdeviceptr).
    pub fn new(mem_alignment: usize, stream: CUstream) -> Self {
        Self {
            memory: HashMap::new(),
            deallocations: Vec::new(),
            stream,
            ptr_bindings: PtrBindings::new(None),
            mem_alignment,
        }
    }

    /// Actually deallocates buffers tagged to be deallocated.
    pub fn perform_deallocations(&mut self) {
        for id in self.deallocations.drain(..) {
            if let Some(ptr) = self.memory.remove(&id) {
                unsafe {
                    cudarc::driver::result::free_async(ptr, self.stream).unwrap();
                }
            }
        }
    }
}

/// The memory resource that can be allocated for CUDA.
#[derive(new, Debug)]
pub struct CudaResource {
    /// The wgpu buffer.
    pub ptr: u64,
    pub binding: *mut std::ffi::c_void,
    offset: u64,
    size: u64,
}

unsafe impl Send for CudaResource {}

pub type Binding = *mut std::ffi::c_void;

impl CudaResource {
    /// Return the binding view of the buffer.
    pub fn as_binding(&self) -> Binding {
        self.binding
    }

    /// Return the buffer size.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Return the buffer offset.
    pub fn offset(&self) -> u64 {
        self.offset
    }
}

impl ComputeStorage for CudaStorage {
    type Resource = CudaResource;
    fn alignment(&self) -> usize {
        self.mem_alignment
    }

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let ptr = self.memory.get(&handle.id).unwrap();

        let offset = handle.offset();
        let size = handle.size();
        let ptr = self.ptr_bindings.register(ptr + offset);

        CudaResource::new(
            *ptr,
            ptr as *const cudarc::driver::sys::CUdeviceptr as *mut std::ffi::c_void,
            offset,
            size,
        )
    }

    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        let id = StorageId::new();
        let ptr = unsafe { cudarc::driver::result::malloc_async(self.stream, size as usize) };
        let ptr = match ptr {
            Ok(ptr) => ptr,
            // I don't think this actually triggers immediately, might be returning the error on the next call
            // Need to figure out how to handle this
            Err(DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY)) => {
                Err(IoError::BufferTooBig(size as usize))?
            }
            Err(other) => panic!("{other}"),
        };
        self.memory.insert(id, ptr);
        Ok(StorageHandle::new(
            id,
            StorageUtilization { offset: 0, size },
        ))
    }

    fn dealloc(&mut self, id: StorageId) {
        self.deallocations.push(id);
    }

    fn flush(&mut self) {
        self.perform_deallocations();
    }
}

/// Physical memory handle.
/// In the current implementation all handles are the same size.
/// PyTorch uses two types of handles: smaller (2MB, more or less the handle size i am aiming for)
/// and bigger handles of 20MB.
/// 2MB is the GPU memory page size (minimum granularity).
// It is preferred to set the handle size to this amount since it allows for finer-grained handle reuse.
type CudaPhysicalHandle = CUmemGenericAllocationHandle;

/// Physical memory allocator for CUDA handles
pub struct CudaPhysicalAllocator {
    device_id: i32,
    memory: HashMap<StorageId, CudaPhysicalHandle>,
    deallocations: VecDeque<CudaPhysicalHandle>,
    granularity: usize
}

impl CudaPhysicalAllocator {
    pub fn new(device_id: i32, granularity: usize) -> Self {


        Self {
            device_id,
            memory: HashMap::new(),
            deallocations: VecDeque::new(),
            granularity
        }
    }
}

impl ComputeStorage for CudaPhysicalAllocator {
    type Resource = CudaPhysicalHandle;

    fn alignment(&self) -> usize {
        // Physical handles are aligned to handle_size
        self.granularity
    }

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        // Return the physical handle for the given storage ID
        *self.memory.get(&handle.id).expect("Invalid storage handle")
    }

    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {

        let size = size.next_multiple_of(self.granularity as u64); // makes sure the allocations are always multiples of gpu page size.
        // Try reuse an unmapped handle first
        let cuda_handle = if let Some(reused_handle) = self.deallocations.pop_front() {
            reused_handle
        } else {
            // Create new handle
            self.create_physical_handle(size)?
        };

        let id = StorageId::new();
        self.memory.insert(id, cuda_handle);

        Ok(StorageHandle::new(
            id,
            StorageUtilization {
                offset: 0,
                size, // Physical handles should have fixed size, but this must be managed by the VirtualStorage
            },
        ))
    }

    fn dealloc(&mut self, id: StorageId) {
        if let Some(cuda_handle) = self.memory.remove(&id) {
            self.deallocations.push_back(cuda_handle);
        }
    }

    fn flush(&mut self) {
        for physical_handle in self.deallocations.drain(..) {
            unsafe {
                cuMemRelease(physical_handle);
            }
        }
    }
}

impl CudaPhysicalAllocator {
    /// Creates a new CUDA physical memory handle
    fn create_physical_handle(&self, size: u64) -> Result<CudaPhysicalHandle, IoError> {
        let mut handle = 0;
        let handle_type = {
            #[cfg(unix)]
            {
                CUmemAllocationHandleType_enum::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
            }
            #[cfg(target_os = "windows")]
            {
                CUmemAllocationHandleType_enum::CU_MEM_HANDLE_TYPE_WIN32
            }
        };

        let prop = CUmemAllocationProp {
            type_: CU_MEM_ALLOCATION_TYPE_PINNED,
            requestedHandleTypes: handle_type,
            location: CUmemLocation {
                type_: CU_MEM_LOCATION_TYPE_DEVICE,
                id: self.device_id,
            },
            win32HandleMetaData: std::ptr::null_mut(),
            allocFlags: Default::default(),
        };

        if unsafe { cuMemCreate(&mut handle, size as usize, &prop, 0).result() }
            .is_err()
        {
            return Err(IoError::BufferTooBig(size as usize));
        }

        Ok(handle)
    }
}

/// CUDA virtual memory mapper implementation
pub struct CudaVirtualMemoryManager {
    device_id: i32,
    base_addr: CUdeviceptr,
    virtual_size: u64,
    handle_size: u64,
    /// Free virtual pages available for allocation
    /// Must stay sorted to guarantee no memory fragmentation happens
    free_pages: Vec<CUdeviceptr>,
}

impl CudaVirtualMemoryManager {
    pub fn new(virtual_size: u64, handle_size: u64, granularity: u64, device_id: i32) -> Self {
        let mut base_addr = 0;
        let handle_size = handle_size.next_multiple_of(granularity);
        let virtual_size = virtual_size.next_multiple_of(granularity);
        let max_pages = virtual_size.div_ceil(handle_size);

        if let Err(e) = unsafe {
            cuMemAddressReserve(
                &mut base_addr,
                virtual_size as usize,
                0, // third argument is the preferred address to reserve the full memory range.
                // If we set 0, CUDA will automatically choose the start address for us.
                0,
                0,
            )
            .result()
        } {
            panic!("Error while attempting to reserve memory: {e}");
        }

        let free_pages: Vec<CUdeviceptr> = (0..max_pages)
            .map(|i| base_addr + i * handle_size)
            .collect();
        Self {
            device_id,
            base_addr,
            virtual_size,
            handle_size,
            free_pages,
        }
    }

    /// Sets memory access permissions for the mapped range
    fn set_access_permissions(&self, addr: u64, size: u64, device_id: i32) {
        let access_desc = CUmemAccessDesc {
            location: CUmemLocation {
                type_: CU_MEM_LOCATION_TYPE_DEVICE,
                id: device_id,
            },
            flags: CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        };

        unsafe { cuMemSetAccess(addr, size as usize, &access_desc, 1) };
    }
}

impl VirtualMemoryManager for CudaVirtualMemoryManager {
    type VirtualAddress = CUdeviceptr;
    type PhysicalHandle = CudaPhysicalHandle;

    fn reserve_range(&mut self, pages: usize) -> Result<Self::VirtualAddress, IoError> {
        if pages == 0 {
            return Err(IoError::InvalidHandle);
        }

        // If we don't have enough free pages, fail the allocation
        if pages > self.free_pages.len() {
            return Err(IoError::BufferTooBig(pages * self.handle_size as usize));
        }

        // Look for a contiguous range in the free pages vector
        for i in 0..=self.free_pages.len() - pages {
            let base = self.free_pages[i];
            let mut is_contiguous = true;

            // Verify the subsequent addresses are contiguous
            for j in 1..pages {
                if self.free_pages[i + j] != base + (j as u64 * self.handle_size) {
                    is_contiguous = false;
                    break;
                }
            }

            if is_contiguous {
                // Remove the pages from the vector (backwards to maintain indices)
                for j in (0..pages).rev() {
                    self.free_pages.remove(i + j);
                }
                return Ok(base);
            }
        }

        // No contiguous range found
        Err(IoError::BufferTooBig(pages * self.handle_size as usize))
    }

    fn release_range(&mut self, addr: Self::VirtualAddress, pages: usize) {
        // Add each page back to free list
        for i in 0..pages {
            let page_addr = addr + (i as u64 * self.handle_size);
            match self.free_pages.binary_search(&page_addr) {
                Ok(_) => {} // Already exists, no-op
                Err(pos) => self.free_pages.insert(pos, page_addr),
            }
        }
    }

    fn page_size(&self) -> u64 {
        self.handle_size
    }

    fn map_handle(
        &mut self,
        virtual_addr: Self::VirtualAddress,
        physical_handle: Self::PhysicalHandle
    ) -> Result<(), IoError> {


        unsafe {
            let res = cuMemMap(
                virtual_addr,
                self.handle_size as usize,
                0, // offset in the handle
                physical_handle,                     // CUmemGenericAllocationHandle
                0,                                   // flags
            );
            if res != CUDA_SUCCESS {
                return Err(IoError::InvalidHandle);
            }
        }

        // Set access permissions
        self.set_access_permissions(virtual_addr, self.handle_size, self.device_id);
        Ok(())
    }

    fn unmap_handle(&mut self, virtual_addr: Self::VirtualAddress) -> Result<(), IoError> {
        unsafe {
            let res = cuMemUnmap(virtual_addr, self.handle_size as usize);
            if res != CUDA_SUCCESS {
                return Err(IoError::InvalidHandle);
            }
        }

        Ok(())
    }
}

impl Drop for CudaVirtualMemoryManager {
    fn drop(&mut self) {
        unsafe {
            cuMemAddressFree(self.base_addr, self.virtual_size as usize);
        }
    }
}

unsafe impl Send for CudaVirtualMemoryManager {}

/// CUDA Virtual Storage that orchestrates physical allocation and virtual mapping
pub struct CudaVirtualStorage {
    /// Physical memory allocator
    physical_allocator: CudaPhysicalAllocator,
    /// Virtual memory manager (handles both address management and mapping)
    virtual_manager: CudaVirtualMemoryManager,
    /// Memory alignment requirement
    granularity: usize, // Typically we will set handle size to this number
    /// Helper for managing GPU kernel parameter bindings
    ptr_bindings: PtrBindings,
    /// Maps storage IDs to their mapped virtual addresses
    /// The Storage id here points to the first physical handle of this memory block.
    memory: HashMap<StorageId, (VirtualBlock, Option<CUdeviceptr>)>,
}

impl CudaVirtualStorage {
    pub fn new(device_id: i32, virtual_size: u64,  handle_size: u64) -> Self {
        // I think this makes more sense here.
        let mut granularity: usize = 0;

        // Handle type will differ by platform
        let handle_type = {
            #[cfg(unix)]
            {
                cudarc::driver::sys::CUmemAllocationHandleType_enum::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
            }
            #[cfg(target_os = "windows")]
            {
                cudarc::driver::sys::CUmemAllocationHandleType_enum::CU_MEM_HANDLE_TYPE_WIN32
            }
        };

        // Define CUDA allocation properties for pinned, device-local memory.
        // Sharing mappings accross devices is still not implemented on [`VirtualStorage`].
        let prop = CUmemAllocationProp {
            type_: CU_MEM_ALLOCATION_TYPE_PINNED,
            requestedHandleTypes: handle_type,
            location: CUmemLocation {
                type_: CU_MEM_LOCATION_TYPE_DEVICE,
                id: device_id,
            },
            win32HandleMetaData: std::ptr::null_mut(),
            allocFlags: Default::default(),
        };

        // Query allocation granularity (GPU page size)
        unsafe {
            cuMemGetAllocationGranularity(
                &mut granularity,
                &prop,
                cudarc::driver::sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            )
        };

        Self {
            physical_allocator: CudaPhysicalAllocator::new(device_id, granularity),
            virtual_manager: CudaVirtualMemoryManager::new(
                virtual_size,
                handle_size,
                granularity as u64,
                device_id,
            ),
            granularity: granularity,
            ptr_bindings: PtrBindings::new(None),
            memory: HashMap::new(),
        }
    }
}

impl VirtualStorage for CudaVirtualStorage {
    type Resource = CudaResource;

    fn alignment(&self) -> usize {
        self.granularity
    }

    fn get(&mut self, id: StorageId) -> Self::Resource {
        let (block, virtual_addr) = self
            .memory
            .get(&id)
            .expect("Invalid storage handle");

        let virtual_addr = virtual_addr.expect("This block is not mapped yet. Please make sure that the block is mapped to a virtual memory range before calling this method.");

        let offset = block.offset();
        let size = block.size();
        let ptr = self.ptr_bindings.register(virtual_addr + offset);

        CudaResource::new(
            *ptr,
            ptr as *const CUdeviceptr as *mut std::ffi::c_void,
            offset,
            size,
        )
    }

    fn alloc(&mut self, size: u64) -> Result<VirtualBlock, IoError> {
        let aligned_size = size.next_multiple_of(self.granularity as u64);
        let handle_size = self.virtual_manager.page_size(); // will return the configured handle size
        let num_handles = aligned_size.div_ceil(handle_size) as usize;

        // Allocate physical handles
        let mut handles = Vec::with_capacity(num_handles);
        for _ in 1..num_handles {
            let handle = self.physical_allocator.alloc(handle_size)?;
            handles.push(handle);
        }

        // Create a new unmapped handle
        Ok(VirtualBlock::new(handles, aligned_size, 0))
    }

    fn map(&mut self, block: &mut VirtualBlock) -> Result<(), IoError> {
        if block.is_mapped() {
            return Err(IoError::InvalidHandle);
        }

        let first_id = block.id();
        let pages = block.pages_needed(self.virtual_manager.page_size());

        // Reserve virtual address space
        let virtual_addr = self.virtual_manager.reserve_range(pages)?;

        // Map each physical handle
        for (i, physical_descriptor) in block.physical_handles.iter().enumerate() {
            let page_addr = virtual_addr + (i as u64 * self.virtual_manager.page_size());
            let cuda_handle = self.physical_allocator.get(physical_descriptor);
            self.virtual_manager
                .map_handle(page_addr, cuda_handle)?;
        }

        // Update state
        block.set_mapped();
        if let Some((stored_block, slot)) = self.memory.get_mut(&first_id) {
            stored_block.set_mapped();
            *slot = Some(virtual_addr);
        }

        Ok(())
    }

    // No op if the block is already unmapped
    fn unmap(&mut self, id: StorageId) {
        if let Some((block, some_addr)) = self.memory.get_mut(&id) {
            if let Some(addr) = some_addr.take() {
                let pages = block.pages_needed(self.virtual_manager.page_size());

                // Unmap each page
                for i in 0..pages {
                    let page_addr = addr + (i as u64 * self.virtual_manager.page_size());
                    let _ = self.virtual_manager.unmap_handle(page_addr);

                }

                for handle in &block.physical_handles {
                    self.physical_allocator.dealloc(handle.id);
                }

                // Release virtual address space
                self.virtual_manager.release_range(addr, pages);
                block.set_unmapped();
            }
        }
    }

    fn dealloc(&mut self, id: StorageId) {
        // First unmap if still mapped
        self.unmap(id);

        // Then deallocate physical handles
        if let Some((block, _)) = self.memory.remove(&id) {
            for descriptor in block.physical_handles {
                self.physical_allocator.dealloc(descriptor.id);
            }
        }
    }

    fn flush(&mut self) {
        self.physical_allocator.flush();
    }
}

unsafe impl Send for CudaVirtualStorage {}


impl Drop for CudaVirtualStorage {
    fn drop(&mut self) {
        // First, unmap and deallocate all virtual blocks
        let block_ids: Vec<StorageId> = self.memory.keys().copied().collect();

        for block_id in block_ids {
            self.dealloc(block_id);
        }

        // Flush any remaining pending deallocations
        self.flush();

    }
}
