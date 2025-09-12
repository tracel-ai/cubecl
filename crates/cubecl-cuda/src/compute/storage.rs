use super::uninit_vec;
use cubecl_core::server::IoError;
use cubecl_runtime::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use cudarc::driver::DriverError;

use cudarc::driver::sys::{
    CUdeviceptr, CUmemAccess_flags_enum::CU_MEM_ACCESS_FLAGS_PROT_READWRITE, CUmemAccessDesc,
    CUmemAllocationHandleType_enum, CUmemAllocationProp,
    CUmemAllocationType_enum::CU_MEM_ALLOCATION_TYPE_PINNED, CUmemGenericAllocationHandle,
    CUmemLocation, CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_DEVICE, CUstream, cuMemAddressFree,
    cuMemAddressReserve, cuMemCreate, cuMemGetInfo_v2, cuMemMap, cuMemRelease, cuMemSetAccess,
    cuMemUnmap, cudaError_enum::CUDA_SUCCESS,
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
enum CudaPhysicalHandle {
    UnmappedHandle {
        handle: CUmemGenericAllocationHandle,
        size: usize,
    },
    MappedHandle {
        handle: CUmemGenericAllocationHandle,
        size: usize,
        address: CUdeviceptr,
    },
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

    /// Maps storage IDs to their mapped set of physical handles, stored in sorted order.
    /// This way, when the storage is asked for a specific virtualblock, it can successfully return the address of the first physical handle.
    memory: HashMap<StorageId, Vec<CudaPhysicalHandle>>,

    /// Helper for managing GPU kernel parameter bindings
    ptr_bindings: PtrBindings,

    /// Free physical handles available for reuse
    deallocations: VecDeque<CudaPhysicalHandle>,
    // Vector of free virtual pages.
    free_pages: VecDeque<CUdeviceptr>,
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
        let max_pages = virtual_size / handle_size; // This storage currently works with fixed size memory pages.

        if let Err(e) =
            unsafe { cuMemAddressReserve(&mut base_addr, virtual_size as usize, 0, 0, 0).result() }
        {
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
            ptr_bindings: PtrBindings::new(None),
            deallocations: VecDeque::new(),
            free_pages: Vec::with_capacity(max_pages)
        }
    }
    /// Creates a new physical memory handle
    fn get_or_create_physical_handle(&mut self, size: u64) -> Result<PhysicalHandle, IoError> {
        // Try reuse an unmapped handle first.
        // As all handles are the same size this is super efficient
        if let Some(handle) = self.deallocations.pop_front() {
            return Ok(handle);
        }

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

        Ok(PhysicalHandle::UnmappedHandle { handle, size })
    }




    /// Sets memory access permissions for the mapped range
    /// [`device_id`] is a parameter to allow for setting access permissions to other devices in the future.
    fn set_access_permissions(&self, addr: CUdeviceptr, size: usize, device_id: usize) {
        let access_desc = CUmemAccessDesc {
            location: CUmemLocation {
                type_: CU_MEM_LOCATION_TYPE_DEVICE,
                id: device_id,
            },
            flags: CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        };

        unsafe { cuMemSetAccess(addr, size, &access_desc, 1) };
    }

    /// Processes all pending deallocations, releasing physical memory.
    fn perform_deallocations(&mut self) {
        for physical_handle in self.deallocations.drain(..) {
            unsafe {
                cuMemRelease(physical_handle.handle);
            }
        }
    }

    fn get_address_for_block(&self, id: StorageId) -> Option<CUdeviceptr> {
        if let Some(mapped_handles) = self.memory.get(id){

        // Runtime check to validate all handles are in mapped state
        assert!(
            mapped_handles
                .iter()
                .all(|h| matches!(h, CudaPhysicalHandle::MappedHandle { .. })),
            "Attempted to get resource but some physical handles are not mapped"
        );

            // Get the base address of the block, which is the address of the first physical handle.
            let base_addr = if let CudaPhysicalHandle::MappedHandle { address, .. } = &mapped_handles[0]
            {
                *address
            } else {
            unreachable!()
            };
            base_addr
        }else
        {
            None
        }

    }



    /// Virtual memory allocation methods.
    /// Allocates a contiguous range of virtual memory addresses.
    /// Will return None if it cannot find a range of contiguous addresses of enough size.
     fn alloc_virtual(&mut self, page_count: u64) -> Option<u64> {
        if page_count == 0 || page_count as usize > self.free_pages.len() {
            return None;
        }

        // Look for a contiguous range in the free pages vector
        for i in 0..=self.free_pages.len() - page_count as usize {
            let base = self.free_pages[i];
            let mut is_contiguous = true;

            // Verify the subsequent addresses are contiguous
            for j in 1..page_count as usize {
                if self.free_pages[i + j] != base + (j as u64 * self.handle_size) {
                    is_contiguous = false;
                    break;
                }
            }

            if is_contiguous {
                // Remove the pages from the vector (backwards)
                for j in (0..page_count as usize).rev() {
                    self.free_pages.remove(i + j);
                }
                return Some(base);
            }
        }

        None
    }

    /// Returns a contiguous virtual memoy range back to the list of free pages.
    fn dealloc_virtual(&mut self, addr: u64) {
        // Insert on correct position to keep the order.
        match self.free_pages.binary_search(&addr) {
                Ok(_) => {}, // Already exists, no-op
                Err(pos) => self.free_pages.insert(pos, addr),
        }

    }
}

impl VirtualStorage for CudaVirtualStorage {
    type Resource = CudaResource;

    fn alignment(&self) -> usize {
        self.mem_alignment
    }

    fn get(&mut self, handle: &VirtualHandle) -> Self::Resource {
        let mapped_handles = self.memory.get(&handle.id).unwrap();

        // Runtime check to validate all handles are in mapped state
        assert!(
            mapped_handles
                .iter()
                .all(|h| matches!(h, CudaPhysicalHandle::MappedHandle { .. })),
            "Attempted to get resource but some physical handles are not mapped"
        );

        // Get the base address of the block, which is the address of the first physical handle.
        let base_addr = if let CudaPhysicalHandle::MappedHandle { address, .. } = &mapped_handles[0]
        {
            *address
        } else {
            unreachable!()
        };

        let offset = handle.offset();
        let size = handle.size();
        let ptr = self.ptr_bindings.register(base_addr + offset);

        CudaResource::new(
            *ptr,
            ptr as *const cudarc::driver::sys::CUdeviceptr as *mut std::ffi::c_void,
            offset,
            size,
        )
    }

    /// Allocates a physical block of the specified size.
    /// Does not perform any mapping.
    fn alloc(&mut self, size: u64) -> Result<VirtualHandle, IoError> {
        let id = StorageId::new();

        let virtual_size = size.next_multiple_of(self.granularity);
        let num_handles = virtual_size / self.handle_size;

        let block_handles = Vec::with_capacity(num_handles);
        for _ in 0..num_handles {
            // Create physical handle
            let handle = self.get_or_create_physical_handle(size)?;
            block_handles.push(handle);
        }

        self.memory.insert(id, block_handles);
        Ok(id)
    }

    /// Sets a physical block for deallocation.
    /// Unmaps all handles and pushes them to the deallocation queue
    fn dealloc(&mut self, id: StorageId) {
        if let Some(handles) = self.memory.remove(&id) {
            for handle in handles {
                match handle {
                    CudaPhysicalHandle::MappedHandle {
                        handle,
                        size,
                        address,
                    } => {
                        // Handle is mapped.
                        // Unmap the handle and push to deallocation queue.
                        unsafe {
                            // Unmap the mapped memory address
                            let res = cuMemUnmap(address, size);
                            if res != CUDA_SUCCESS {
                                panic!("cuMemUnmap failed: {:?}", res);
                            }
                        }
                        self.dealloc_virtual(address);
                        // Return the handle for future reuse.
                        self.deallocations
                            .push_back(CudaPhysicalHandle::UnmappedHandle { handle, size });
                    }
                    CudaPhysicalHandle::UnmappedHandle { handle, size } => {
                        // Already unmapped, just push queue for future reuse.
                        self.deallocations
                            .push_back(CudaPhysicalHandle::UnmappedHandle { handle, size });
                    }
                }
            }
        }
    }

    /// Flushes all pending deallocations
    fn flush(&mut self) {
        self.perform_deallocations();
    }

    /// Maps enough physical handles to form a contiguous virtual block of the specified size.
    fn map(&mut self, handle: &mut VirtualHandle) -> Result<(), IoError> {
        let id = handle.id();

        let aligned_size = handle.size().next_multiple_of(self.granularity);
        assert!(handle.is_unmapped, "Cannot re-map an already mapped handle!");
        // The required handles of this block should be in memory.
        let physical_handles = self
        .memory
        .get_mut(&id)
        .expect("Invalid storage ID in map");

        let num_handles = physical_handles.len();

        let base_addr = self.alloc_virtual(num_handles).unwrap_or_else({
            return Err(IoError::BufferTooBig(aligned_size as usize));
        })


        for (i, ph) in physical_handles.iter_mut().enumerate() {
            let virt_addr = base_addr + (i as u64) * self.handle_size;

            match ph {
                CudaPhysicalHandle::UnmappedHandle { handle, size } => {
                    unsafe {
                        let res = cuMemMap(
                            virt_addr,
                            *size,
                            0,       // offset in the handle
                            *handle, // CUmemGenericAllocationHandle
                            0,       // flags
                        );
                        if res != CUDA_SUCCESS {
                            return Err(IoError::InvalidHandle);
                        }
                    }

                    // Set access permissions
                    self.set_access_permissions(virt_addr, *size, self.device_id as usize);

                    // Update the entry to create a mapped handle.
                    *ph = CudaPhysicalHandle::MappedHandle {
                        handle: *handle,
                        size: *size,
                        address: virt_addr,
                    };
                }
                CudaPhysicalHandle::MappedHandle { .. } => {
                    /// This would indicate memory corruption
                    panic!("Handle already mapped for StorageId {:?}", id);
                }
            }
        }



    }

    /// Unmaps all handles associated with the given storage ID.
/// Leaves the handles in memory (still owned by this allocator) but in unmapped state,
/// ready for reuse or eventual deallocation.
fn unmap(&mut self, id: StorageId) {


    if let Some(handles) = self.memory.get_mut(&id) {



        for ph in handles.iter_mut() {
            match ph {
                CudaPhysicalHandle::MappedHandle {
                    handle,
                    size,
                    address,
                } => {
                    // Call unmap over the virtual address
                    unsafe {
                        let res = cuMemUnmap(*address, *size);
                        if res != CUDA_SUCCESS {
                            panic!("cuMemUnmap failed for StorageId {:?}: {:?}", id, res);
                        }
                    }

                    self.dealloc_virtual(address);

                    // Convert from mapped to unmapped
                    *ph = CudaPhysicalHandle::UnmappedHandle {
                        handle: *handle,
                        size: *size,
                    };
                }
                CudaPhysicalHandle::UnmappedHandle { .. } => {
                    //  No-op in this case. The handle is already unmapped
                }
            }
        }
        }
    }



}

impl Drop for CudaVirtualStorage {
    fn drop(&mut self) {

        // Flush all pending deallocations
        self.perform_deallocations();

        // Unmap all blocks that are in memory
        for (_id, handles) in self.memory.drain() {
            for ph in handles {
                match ph {
                    CudaPhysicalHandle::MappedHandle {
                        handle,
                        size,
                        address,
                    } => {
                        // First unmap
                        unsafe {
                            let _ = cuMemUnmap(address, size);
                        }
                        self.dealloc_virtual(address);
                        // Then release the physical handle
                        unsafe {
                            cuMemRelease(handle);
                        }
                    }
                    CudaPhysicalHandle::UnmappedHandle { handle, .. } => {
                        //  If already unmapped just release the handle.
                        unsafe {
                            cuMemRelease(handle);
                        }
                    }
                }
            }
        }

        // Free virtual address space
        unsafe {
            cuMemAddressFree(self.base_addr, self.virtual_size as usize);
        }
    }
}

unsafe impl Send for CudaVirtualStorage {}
