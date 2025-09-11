use super::uninit_vec;
use cubecl_core::server::IoError;
use cubecl_runtime::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use cudarc::driver::DriverError;

use cudarc::driver::sys::{
    CUdeviceptr, CUmemAccess_flags_enum::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    CUmemAccessDesc, CUmemAllocationHandleType_enum, CUmemAllocationProp,
    CUmemAllocationType_enum::CU_MEM_ALLOCATION_TYPE_PINNED, CUmemGenericAllocationHandle,
    CUmemLocation, CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_DEVICE, CUstream,
    cuMemAddressFree, cuMemAddressReserve, cuMemCreate, cuMemGetInfo_v2, cuMemMap, cuMemRelease,
    cuMemSetAccess, cuMemUnmap, cudaError_enum::CUDA_SUCCESS,
};

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
#[derive(Debug)]
enum PhysicalHandle {

    UnmappedHandle {
        handle: CUmemGenericAllocationHandle,
        size: usize
    },
    MappedHandle {
        handle: CUmemGenericAllocationHandle,
        size: usize
        address: CUdeviceptr
    }
}


/// [`CudaVirtualStorage`] implementing the [`VirtualStorage`] trait
pub struct CudaVirtualStorage {
    /// CUDA device ID for this storage instance
    device_id: i32,
    /// CUDA stream for asynchronous memory operations
    stream: CUstream,
    /// Base address of the reserved virtual address space
    base_addr: CUdeviceptr,
    /// Next available virtual address for new allocations
    next_addr: CUdeviceptr,
    /// Total size of the reserved virtual address space
    virtual_size: u64,
    /// Size of each physical memory handle (aligned to granularity)
    handle_size: u64,
    /// Memory alignment requirement for allocations
    mem_alignment: usize,

    /// Maps storage IDs to their mapped virtual addresses
    memory: HashMap<StorageId, PhysicalHandle>,

    /// Helper for managing GPU kernel parameter bindings
    ptr_bindings: PtrBindings,

    /// Free physical handles available for reuse
    unmapped_handles: VecDeque<PhysicalHandle>,
    // Unmapped physical handles which are pending deallocation
    deallocations: Vec<StorageId>
}

impl CudaVirtualStorage {
    /// Creates a new virtual storage allocator.
    pub fn new(
        device_id: i32,
        stream: CUstream,
        virtual_size: u64,
        alignment: u64,
        handle_size: u64,
    ) -> Self {
        let mut base_addr = 0;
        let handle_size = handle_size.next_multiple_of(alignment);
        let virtual_size = virtual_size.next_multiple_of(alignment);

        if let Err(e) = unsafe {
            cuMemAddressReserve(&mut base_addr, virtual_size as usize, 0, 0, 0).result()
        } {
            panic!("Error while attempting to reserve memory: {e}");
        };

        Self {
            device_id,
            stream,
            base_addr,
            next_addr: base_addr,
            virtual_size,
            handle_size,
            mem_alignment: handle_size as usize,
            physical_handles: HashMap::new(),
            mapped_blocks: HashMap::new(),
            deallocations: Vec::new(),
            ptr_bindings: PtrBindings::new(None),
            free_physical_handles: VecDeque::new(),
        }
    }
    /// Creates a new physical memory handle
    fn create_physical_handle(&mut self, size: u64) -> Result<PhysicalHandle, IoError> {
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

        if unsafe { cuMemCreate(&mut handle, size as usize, &prop, 0).result() }.is_err() {
            return Err(IoError::BufferTooBig(size as usize));
        }

        Ok(PhysicalHandle::UnmappedHandle{
            handle,
            size
        })
    }

    /// Sets memory access permissions for the mapped range
    /// `device_id` is a parameter to allow for setting access permissions to other devices in the future.
    fn set_access_permissions(&self, addr: CUdeviceptr, size: usize, device_id: usize) {
        let access_desc = CUmemAccessDesc {
            location: CUmemLocation {
                type_: CU_MEM_LOCATION_TYPE_DEVICE,
                id: device_id,
            },
            flags: CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        };

        let _ = unsafe { cuMemSetAccess(addr, size, &access_desc, 1).result() };
    }

    /// Processes all pending deallocations
    fn perform_deallocations(&mut self) {
        // At this point all deallocated handles.
        for id in self.deallocations.drain(..) {
            if let Some(handle) = self.memory.remove(&id) {
               // Need a way to remove handles from here.
            }
        }
    }

    /// Gets an available physical handle, reusing a free one or creating a new one
    // TODO: REVIEW THIS
    fn get_physical_handle(&mut self) -> Result<StorageId, IoError> {
        // Try to reuse a free physical handle first
        if let Some(free_id) = self.free_physical_handles.pop_front() {
            if let Some(handle) = self.physical_handles.get_mut(&free_id) {
                return Ok(free_id);
            }
        }

        // No free handle available, create a new one
        self.alloc(self.handle_size)
    }
}

impl VirtualStorage for CudaVirtualStorage {
    type Resource = CudaResource;

    fn alignment(&self) -> usize {
        self.mem_alignment
    }

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let mapped_block = self.mapped_blocks.get(&handle.id).unwrap();
        let ptr = mapped_block.virtual_block.get_ptr();

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

    /// Allocates a physical block of the specified size.
    /// Does not perform any mapping.
    fn alloc(&mut self, size: u64) -> Result<StorageId, IoError> {
        let id = StorageId::new();

        // Create physical handle
        let handle = self.create_physical_handle(size)?;

        // Store physical handle info
        let physical_handle = PhysicalHandle {
            handle,
            size,
            is_free: false,
        };

        self.physical_handles.insert(id, physical_handle);

        Ok(id)
    }

    /// Marks a physical block for deallocation
    fn dealloc(&mut self, id: StorageId) {
        self.deallocations.push(id);
    }

    /// Flushes all pending deallocations
    fn flush(&mut self) {
        self.perform_deallocations();
    }

    /// Maps enough physical handles to form a contiguous virtual block of the specified size.
    /// Returns a StorageHandle that can be used to access the mapped memory.
    fn map(&mut self, size: usize) -> Result<StorageHandle, IoError> {
        let aligned_size = (size as u64).next_multiple_of(self.handle_size);
        let num_handles_needed = (aligned_size / self.handle_size) as usize;

        // Calculate virtual address for this mapping
        let virtual_addr = self.next_addr;

        // Create virtual block
        let mut virtual_block = VirtualBlock::from_reserved(
            virtual_addr,
            aligned_size,
            self.handle_size
        );

        // Get the required physical handles
        let mut physical_handle_ids = Vec::with_capacity(num_handles_needed);
        for _ in 0..num_handles_needed {
            let handle_id = self.get_physical_handle()?;
            physical_handle_ids.push(handle_id);
        }

        // Map each physical handle to its corresponding virtual address
        for (i, &handle_id) in physical_handle_ids.iter().enumerate() {
            let physical_handle = self.physical_handles.get(&handle_id).unwrap();
            let virtual_addr_offset = virtual_addr + (i as u64 * self.handle_size);

            if unsafe {
                cuMemMap(
                    virtual_addr_offset,
                    self.handle_size as usize,
                    0,
                    physical_handle.handle,
                    0,
                ).result()
            }.is_err() {
                // If mapping fails, cleanup any successful mappings
                for j in 0..i {
                    let cleanup_addr = virtual_addr + (j as u64 * self.handle_size);
                    let _ = unsafe {
                        cuMemUnmap(cleanup_addr, self.handle_size as usize).result()
                    };
                }
                // Return physical handles to free pool
                for &handle_id in &physical_handle_ids {
                    if let Some(handle) = self.physical_handles.get_mut(&handle_id) {
                        handle.is_free = true;
                        self.free_physical_handles.push_back(handle_id);
                    }
                }
                return Err(IoError::InvalidHandle);
            }

            // Set access permissions for this handle
            self.set_access_permissions(virtual_addr_offset, self.handle_size as usize);
        }

        // Update virtual block state
        virtual_block.set_mapped();

        // Create a new storage ID for this mapped block
        let block_id = StorageId::new();
        let mapped_block = MappedBlock {
            virtual_block,
            physical_handles: physical_handle_ids,
        };

        self.mapped_blocks.insert(block_id, mapped_block);

        // Advance next address
        self.next_addr = virtual_addr + aligned_size;

        Ok(StorageHandle::new(
            block_id,
            StorageUtilization {
                offset: 0,
                size: size as u64
            },
        ))
    }

    /// Unmaps all handles associated with the given storage ID.
    fn unmap(&mut self, id: StorageId) {
        if let Some(mapped_block) = self.mapped_blocks.remove(&id) {
            let virtual_block = &mapped_block.virtual_block;

            // Unmap each handle in the block
            for (i, &physical_id) in mapped_block.physical_handles.iter().enumerate() {
                let virtual_addr_offset = virtual_block.base_addr + (i as u64 * self.handle_size);

                // Unmap the virtual memory
                let _ = unsafe {
                    cuMemUnmap(virtual_addr_offset, self.handle_size as usize).result()
                };

                // Mark physical handle as free for reuse
                if let Some(physical_handle) = self.physical_handles.get_mut(&physical_id) {
                    physical_handle.is_free = true;
                    self.free_physical_handles.push_back(physical_id);
                }
            }
        }
    }
}

impl Drop for CudaVirtualStorage {
    fn drop(&mut self) {
        self.flush();

        // Unmap all mapped blocks
        for (_, mapped_block) in &self.mapped_blocks {
            let virtual_block = &mapped_block.virtual_block;
            if virtual_block.state == BlockState::Mapped {
                for i in 0..mapped_block.physical_handles.len() {
                    let virtual_addr_offset = virtual_block.base_addr + (i as u64 * self.handle_size);
                    unsafe {
                        cuMemUnmap(virtual_addr_offset, self.handle_size as usize);
                    }
                }
            }
        }

        // Release all physical handles
        for (_, physical_handle) in self.physical_handles.drain() {
            unsafe {
                cuMemRelease(physical_handle.handle);
            }
        }

        // Free virtual address space
        unsafe {
            cuMemAddressFree(self.base_addr, self.virtual_size as usize);
        }
    }
}

unsafe impl Send for CudaVirtualStorage {}
