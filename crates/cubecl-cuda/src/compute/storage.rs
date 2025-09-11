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

    #[cfg(test)]
    pub fn deallocations_len(&self) -> usize {
        self.deallocations.len()
    }

    #[cfg(test)]
    pub fn memory_len(&self) -> usize {
        self.memory.len()
    }

    #[cfg(test)]
    pub fn memory_contains_key(&self, id: &StorageId) -> bool {
        self.memory.contains_key(id)
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




// I decided to implement this simple FIFO pool instead of creating a storage that returns handles and create a new pool in the memory manage module because I think it fits more with your current structure.
// The goal of the pool is to automatically manage physical memory reusability. This way when blocks are deallocated by the pool, they are just unmapped, but memory is not completely released until pool goes out of scope or you call the [`flush`] method inside the pool.
// Think of it as a data structure owned by the [`VirtualStorage`] that just holds the physical memory, which when working with VMM, must be separated from virtual memory (represented as [`VirtualBlocks`] in my implementation)
struct HandlePool {
    device_id: i32, // Device id for setting access permissions

    max_handles: u64, // Max number of handles that can fit in this pool.
    /// Size of each physical memory handle in bytes.
    /// Should be a multiple of the device's allocation granularity.
    handle_size: u64, // Size of each handle. Handle size is fixed for the whole pool.

    // Growing map of cuda handles, mapped to virtual memory addresses.
    allocated_handles: HashMap<CUdeviceptr, CUmemGenericAllocationHandle>,
    // Free FIFO queue of free handles.
    free_queue: VecDeque<CUmemGenericAllocationHandle>,
}

impl HandlePool {
    /// Creates a new handle pool for the specified device.
    ///
    /// # Arguments
    ///
    /// * `device_id` - CUDA device ID where handles will be allocated
    /// * `handle_size` - Size of each handle in bytes (must be multiple of granularity)
    /// * `max_size` - Maximum total memory this pool can manage
    ///   NOTE: Assumes handle size has already been set to a multiple of the minimum allocation granularity of the target device.
    fn new(device_id: i32, handle_size: u64, max_size: u64) -> Self {

        let max_handles = max_size / handle_size;

        Self {
            device_id,
            max_handles,
            handle_size,

            allocated_handles: HashMap::new(),
            free_queue: VecDeque::new(),
        }
    }

    // Utility to check the device current memory usage.
    // Returns a tuple with the free memory and the total available memory in bytes.
    fn get_device_memory_info() -> Result<(u64, u64), DriverError> {
        // Consultar memoria
        let mut free_bytes = 0usize;
        let mut total_bytes = 0usize;

        unsafe {
            cuMemGetInfo_v2(&mut free_bytes, &mut total_bytes).result()?;
        }
        Ok((free_bytes as u64, total_bytes as u64))
    }

    /// Attempts to map a physical handle to a virtual address.
    ///
    /// This function performs the actual mapping operation and sets up memory
    /// access permissions for the device.
    ///
    /// # Arguments
    ///
    /// * `new_addr` - Virtual address where the handle should be mapped
    /// * `handle` - Physical memory handle to map
    ///
    /// # Returns
    ///
    /// Returns the virtual address on success, or `IoError::InvalidHandle` if
    /// mapping fails.
    fn try_map_handle(&mut self, new_addr: CUdeviceptr, handle: CUmemGenericAllocationHandle) -> Result<CUdeviceptr,IoError>{

            if unsafe{cuMemMap(new_addr, self.handle_size as usize, 0, handle, 0).result()}.is_err()
            {

                Err(IoError::InvalidHandle)

            } else {
                // Set the access permissions for this device.
                self.set_access_permissions(new_addr, self.device_id);
                self.allocated_handles.insert(new_addr, handle);
                Ok(new_addr)
            }
     }


    /// Allocates a physical memory handle and maps it to a virtual address.
    ///
    /// This function implements a two-tier allocation strategy:
    /// 1. First, try to reuse a handle from the free queue (FIFO)
    /// 2. If no free handles, create a new physical memory handle
    /// 3. If creation fails due to OOM, attempt eviction and retry
    ///
    /// # Arguments
    ///
    /// * `new_addr` - Virtual address where the handle should be mapped
    /// * from_eviction - If this allocation attempt comes from a previous eviction.
    ///
    /// # Returns
    ///
    /// Returns the virtual address on success.
    ///
    /// # Errors
    ///
    /// * `IoError::InvalidHandle` - If mapping fails
    /// * `IoError::BufferTooBig` - If no memory available and eviction impossible, or if the allocation came from a previous eviction.
    fn allocate_handle(&mut self, new_addr: CUdeviceptr, from_eviction: bool) -> Result<CUdeviceptr, IoError> {
        // First, we attempt to pop from the free queue in a FIFO way.
        // The first handle that was deallocated is the first that is reused.
        if let Some(free_handle) = self.free_queue.pop_front() {


            return self.try_map_handle(new_addr, free_handle);

        }

        let mut handle = 0;
        let handle_type = {
            #[cfg(unix)]
            {
                CUmemAllocationHandleType_enum::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
            }
            #[cfg(target_os = "windows")]
            {
                CU_MEM_HANDLE_TYPE_WIN32
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

        // If physical memory creation fails, we attempt to evict one handle.
        if unsafe{ cuMemCreate(
            &mut handle,
            self.handle_size as usize,
            &prop,
            0
        ).result()}
        .is_err()
        {
            // To avoid infinite recursion, the max number of retries is set to 1.
            if from_eviction {
                return Err(IoError::BufferTooBig(self.handle_size as usize));
            }

            let (free_bytes, _) = Self::get_device_memory_info().unwrap();

            if free_bytes >= self.handle_size && !self.free_queue.is_empty() {
                self.evict(1); // Evict handles from the free pool.
                self.allocate_handle(new_addr, true)?;

            } else {
                return Err(IoError::BufferTooBig(self.handle_size as usize));
            }
        }

        self.try_map_handle(new_addr, handle)
    }



    // Deallocation functions like dealloc and flush do not return errors in CubeCL.
    ///
    /// # Arguments
    ///
    /// * `new_addr` - Virtual address where the handle should be mapped
    ///
    /// # Returns
    ///
    /// Returns the virtual address on success.
    ///
    /// # Errors
    ///
    /// * `IoError::InvalidHandle` - If mapping fails
    /// * `IoError::BufferTooBig` - If no memory available and eviction impossible
    // I keep it as is for compatibility.
    pub fn deallocate_handle(&mut self, addr: CUdeviceptr) {
        if let Some(handle) = self.allocated_handles.remove(&addr) {
            // Note: I am not sure if it is a good idea or not to let return an error here in case of failure on the device side. PyTorch does not ignore unmapping errors (See the macro C10_CUDA_DRIVER_CHECK  at https://github.dev/pytorch/pytorch/blob/main/c10/cuda/CUDACachingAllocator.cpp)
            let _ = unsafe{cuMemUnmap(addr, self.handle_size as usize).result()};
            self.free_queue.push_back(handle);
        }
    }

    /// Reserves capacity for additional handles in internal data structures.
    fn reserve(&mut self, num_handles: usize) -> Result<(), IoError> {
        let current_size = self.allocated_handles.len() + self.free_queue.len();
        if current_size + num_handles > self.max_handles as usize {
            return Err(IoError::BufferTooBig(
                num_handles * self.handle_size as usize,
            ));
        }
        self.allocated_handles.reserve(num_handles);
        Ok(())
    }

    /// Permanently releases physical memory handles from the free queue.
    ///
    /// This function removes handles from the FIFO queue and calls `cuMemRelease()`
    /// to return the physical memory to the CUDA driver.
    ///
    /// # Arguments
    ///
    /// * `n_handles` - Number of handles to evict from the end of the queue
    fn evict(&mut self, n_handles: usize) {
        self.free_queue
            .drain((self.free_queue.len().saturating_sub(n_handles))..)
            .for_each(|handle| {
                let _ = unsafe{cuMemRelease(handle).result()};
            });


    }


    /// Permanently releases all handles in the free queue.
    ///
    /// This operation cannot be undone - all freed physical memory is returned
    /// to the CUDA driver and cannot be reused by this pool.
    fn flush(&mut self) {
        self.free_queue.drain(..).for_each(|handle| {
            let _ = unsafe{cuMemRelease(handle).result()};
        });

        // Clear the free queue
        self.free_queue.clear();
    }

    /// Sets memory access permissions for the mapped range
    ///
    /// After mapping, we need to explicitly grant read/write access to the device.
    /// This is separate from the mapping step in CUDA VMM.
    ///
    /// The parameter [`device_id`] is left there to enable data sharing over RDMA in the future.
    /// Note that this implementation does not include the [`share`] and [`from_shared`] methods found in:
    /// https://github.dev/pytorch/pytorch/blob/main/c10/cuda/CUDACachingAllocator.cpp
    fn set_access_permissions(&self, addr: CUdeviceptr, device_id: i32) {
        let access_desc = CUmemAccessDesc {
            location: CUmemLocation {
                type_: CU_MEM_LOCATION_TYPE_DEVICE,
                id: device_id,
            },
            flags: CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        };

        let _ = unsafe{cuMemSetAccess(
            addr,
            self.handle_size as usize,
            &access_desc,
            1
        ).result()};
    }
}

impl Drop for HandlePool {
    /// Ensures all GPU memory is properly released when the pool is dropped.
    ///
    /// This implementation prevents memory leaks by:
    /// 1. Releasing all allocated physical handles with `cuMemRelease()`
    /// 2. Releasing all free handles via `flush()`
    /// 3. Clearing internal data structures
    fn drop(&mut self) {
        self.allocated_handles.drain().for_each(|(_, handle)| {
            let _ = unsafe{cuMemRelease(handle).result()};
        });
        self.allocated_handles.clear();
        self.flush();
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum BlockState {
    Unmapped, // Virtual space reserved, no physical memory
    Mapped,   // Physical memory mapped, available for allocation
}


/// A contiguous block of virtual memory that can be mapped to physical handles.
///
/// `VirtualBlock` represents a logical allocation unit that may span multiple
/// physical memory handles. The block manages the mapping between a contiguous
/// virtual address range and the underlying physical memory granules.
///
/// The memory layout is the following
///
/// ```text
/// Virtual Block (e.g., 10MB)
/// ├── Handle 0 (2MB) -> Physical Memory A
/// ├── Handle 1 (2MB) -> Physical Memory B
/// ├── Handle 2 (2MB) -> Physical Memory C
/// ├── Handle 3 (2MB) -> Physical Memory D
/// └── Handle 4 (2MB) -> Physical Memory
#[derive(Debug)]
pub struct VirtualBlock {
    state: BlockState,  // Unmapped -> Mapped
    /// Number of physical handles backing this virtual block
    num_handles: usize,
    virtual_size: u64, // Total size of the block.
    base_addr: CUdeviceptr, // Base virtual address or the block.
}

impl VirtualBlock {
    fn from_reserved(base_addr: CUdeviceptr, virtual_size: u64, handle_size: u64) -> Self {
        Self {
            base_addr, // Initialized ptr
            state: BlockState::Unmapped,
            virtual_size,
            num_handles: (virtual_size / handle_size) as usize,
        }
    }

    /// Map the block to mark it as mapped
    fn set_mapped(&mut self) {
        self.state = BlockState::Mapped;
    }


    /// Marks the block as unmapped from physical memory.
    ///
    /// This should be called when physical handles are unmapped but the
    /// virtual address space is retained.
    fn set_unmapped(&mut self) {
        self.state = BlockState::Unmapped;
    }



    pub fn get_ptr(&self) -> CUdeviceptr {
        self.base_addr
    }
}

/// The idea of [`Virtual Storage`] is to reserve a large
/// virtual address (VA) range once and *expand* it on demand by mapping
/// additional physical pages into that range. This avoids repeatedly
/// allocating slightly-larger chunks as shapes fluctuate (e.g. batched
/// inference), which would otherwise create many small, unrecoverable
/// fragments ("slivers").
pub struct VirtualStorage {
     /// CUDA device ID for this storage instance
    device_id: i32,
    /// CUDA stream for asynchronous memory operations
    stream: CUstream,
     /// Base address of the reserved virtual address space
    base_addr: CUdeviceptr,
    /// Next available virtual address for new allocations. Think of it as the program break pointer on standard Bump Allocators
    next_addr: CUdeviceptr,
     /// Total size of the reserved virtual address space
    virtual_size: u64,
     /// Size of each physical memory handle (aligned to granularity)
    handle_size: u64,
    /// Memory alignment requirement for allocations
    mem_alignment: usize,
    /// Internal pool of physical memory handles.
    pool: HandlePool,
    /// Maps storage IDs to their corresponding virtual blocks
    memory: HashMap<StorageId, VirtualBlock>, // All blocks go here

    /// Storage IDs pending deallocation.
    /// Note that i followed this lazy deallocation strategy to match CudaStorage behavior.
    /// Would need more time to figure out if it is the best idea on this case, as virtual memory is not really deallocated, just mapped/unmapped which should be much cheaper that cudaMalloc / cudaFree.
    deallocations: Vec<StorageId>,
    /// Helper for managing GPU kernel parameter bindings
    ptr_bindings: PtrBindings,
}

impl VirtualStorage {

     /// Creates a new virtual storage allocator.
    ///
    /// # Arguments
    ///
    /// * `device_id` - CUDA device ID where memory will be allocated
    /// * `stream` - CUDA stream for async operations
    /// * `virtual_size` - Size of virtual address space to reserve
    /// * `alignment` - Memory alignment requirement
    /// * `handle_size` - Size of each physical memory handle
    ///
    /// # Panics
    ///
    /// Panics if virtual address space reservation fails via `cuMemAddressReserve()`.
    ///
    /// # Memory Layout
    ///
    /// The constructor performs these key operations:
    /// 1. Aligns handle_size and virtual_size to the specified alignment
    /// 2. Reserves a contiguous virtual address space of virtual_size bytes
    /// 3. Initializes the handle pool with the aligned handle size
    /// 4. Sets up internal tracking structures
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


        if let Err(e) = unsafe{cuMemAddressReserve(
            &mut base_addr,
            virtual_size as usize,
            0,
            0,
            0
        ).result()} {
            panic!("Error while attempting to reserve memory: {e}");
        };

        Self {
            device_id,
            stream,
            base_addr,
            next_addr: base_addr, // Initialize the next block's addr to the base addr of the storage.
            virtual_size,
            handle_size,
            mem_alignment: handle_size as usize,
            // Internal pool of handles.
            pool: HandlePool::new(device_id, handle_size, virtual_size),
            memory: HashMap::new(), // All blocks go here
            deallocations: Vec::new(),
            ptr_bindings: PtrBindings::new(None),

        }
    }

    // Returns the next memory address that can be assigned, to ensure that blocks are contiguous in memory.
    fn next_addr(&self) -> CUdeviceptr {
        self.next_addr
    }

    fn set_next_addr(&mut self, next_addr: CUdeviceptr) {
        self.next_addr = next_addr;
    }

    /// Processes all pending deallocations.
    ///
    /// This function implements the lazy deallocation strategy by:
    /// 1. Iterating through all pending deallocation IDs
    /// 2. Unmapping physical handles for each virtual block
    /// 3. Returning physical handles to the pool for reuse
    /// 4. Removing virtual blocks from tracking structures
    ///
    /// # Notes
    ///
    /// Lazy deallocation batches unmap operations to reduce syscall overhead.
    /// However, with virtual memory management, unmapping is relatively fast
    /// compared to traditional memory allocation schemes.
    // Therefore, the lazy deallocation pattern might not be needed here.
    fn perform_deallocations(&mut self) {
        for id in self.deallocations.drain(..) {
            if let Some(block) = self.memory.remove(&id) {
                for i in 0..block.num_handles {
                    let addr = block.base_addr + i as u64 * self.handle_size;
                    self.pool.deallocate_handle(addr);
                }
            }
        }
    }


}

impl ComputeStorage for VirtualStorage {
    type Resource = CudaResource;

    /// Returns the memory alignment requirement.
    fn alignment(&self) -> usize {
        self.mem_alignment
    }

    /// Retrieves a resource for GPU kernel execution.
    ///
    /// This function converts a storage handle into a GPU-accessible resource by:
    /// 1. Looking up the virtual block associated with the storage ID
    /// 2. Computing the effective address with handle offset
    /// 3. Registering the address for GPU kernel parameter binding
    /// 4. Creating a CudaResource with the bound address
    ///
    /// # Arguments
    ///
    /// * `handle` - Storage handle containing ID, offset, and size information
    ///
    /// # Returns
    ///
    /// A `CudaResource` that can be used as a kernel parameter.
    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let block = self.memory.get(&handle.id).unwrap();
        let ptr = block.get_ptr();

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
    /// Allocates a new virtual memory block.
    ///
    /// 1. Determines how many physical handles needed
    /// 2. Pre-allocates space in internal data structures
    /// 3. Allocates and maps each required handle
    /// 4. Marks block as mapped and advances allocation pointer
    /// 5. Adds to internal tracking with new storage ID
    ///
    /// # Arguments
    ///
    /// * `size` - Requested allocation size in bytes
    ///
    /// # Returns
    ///
    /// A `StorageHandle` that can be used to access the allocated memory.
    ///
    /// # Errors
    ///
    /// * `IoError::BufferTooBig` - If insufficient virtual address space
    /// * `IoError::InvalidHandle` - If physical memory allocation fails
    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        let id = StorageId::new();

        let next_addr = self.next_addr();
        let mut block = VirtualBlock::from_reserved(next_addr, size, self.handle_size);

        self.pool.reserve(block.num_handles)?; // Expand the size of the pool first to avoid copying the hashmap at each iteration
        // Allocate handles for this block.
        for i in 0..block.num_handles {
            let addr = block.base_addr + i as u64 * self.handle_size;
            self.pool.allocate_handle(addr, false)?;
        }

        let next_addr = next_addr + block.virtual_size;
        self.set_next_addr(next_addr);

        block.set_mapped(); // All handles are mapped now so we set the block to mapped state
        self.memory.insert(id, block);

        Ok(StorageHandle::new(
            id,
            StorageUtilization { offset: 0, size },
        ))
    }

    /// Marks a storage block for lazy deallocation.
    ///
    /// This function implements CubeCL's lazy deallocation pattern for compatibility,
    /// though with virtual memory management, immediate deallocation would be more
    /// efficient since unmapping is a fast operation.
    ///
    /// # Arguments
    ///
    /// * `id` - Storage ID to deallocate
    fn dealloc(&mut self, id: StorageId) {
        self.deallocations.push(id);
    }

    /// Forces processing of all pending deallocations.
    ///
    /// This function should be called periodically to prevent excessive buildup
    /// of pending deallocations and to reclaim physical memory handles for reuse.
    fn flush(&mut self) {
        self.perform_deallocations()
    }
}

impl Drop for VirtualStorage {
    fn drop(&mut self) {
        self.flush();

        for addr in self.pool.allocated_handles.keys() {
            unsafe{cuMemUnmap(*addr, self.handle_size as usize)};
        }

        self.pool.allocated_handles.drain().for_each(|(_, handle)| {
            unsafe {cuMemRelease(handle)};
        });
        self.pool.allocated_handles.clear();
        self.pool.flush();
        unsafe{cuMemAddressFree(self.base_addr, self.virtual_size as usize)};
    }
}

unsafe impl Send for VirtualStorage {}

// This enum is just to enable storage dispatch at the runtime level
pub enum CudaStorageType {
    Regular(CudaStorage),
    Expandable(VirtualStorage),
}

impl ComputeStorage for CudaStorageType {
    type Resource = CudaResource;

    fn alignment(&self) -> usize {
        match self {
            CudaStorageType::Regular(storage) => storage.alignment(),
            CudaStorageType::Expandable(storage) => storage.alignment(),
        }
    }

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        match self {
            CudaStorageType::Regular(storage) => storage.get(handle),
            CudaStorageType::Expandable(storage) => storage.get(handle),
        }
    }

    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        match self {
            CudaStorageType::Regular(storage) => storage.alloc(size),
            CudaStorageType::Expandable(storage) => storage.alloc(size),
        }
    }

    fn dealloc(&mut self, id: StorageId) {
        match self {
            CudaStorageType::Regular(storage) => storage.dealloc(id),
            CudaStorageType::Expandable(storage) => storage.dealloc(id),
        }
    }

    fn flush(&mut self) {
        match self {
            CudaStorageType::Regular(storage) => storage.flush(),
            CudaStorageType::Expandable(storage) => storage.flush(),
        }
    }
}
