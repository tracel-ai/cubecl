use super::uninit_vec;
use cubecl_core::server::IoError;
use cubecl_runtime::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use cudarc::driver::DriverError;

use cudarc::driver::sys::{
    CUdevice, CUdeviceptr, CUmemAccess_flags_enum::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    CUmemAccessDesc,
    CUmemAllocationHandleType_enum,
    CUmemAllocationProp,
    CUmemAllocationType_enum, CUmemAllocationType_enum::CU_MEM_ALLOCATION_TYPE_PINNED,
    CUmemGenericAllocationHandle, CUmemLocation,
    CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_DEVICE, CUstream, cuDeviceGet, cuMemAddressFree,
    cuMemAddressReserve, cuMemCreate, cuMemGetInfo_v2, cuMemMap,
    cuMemRelease, cuMemSetAccess, cuMemUnmap,  cudaError_enum::CUDA_SUCCESS,
};

use std::collections::{HashMap, VecDeque};



// Type aliases for better readability.
type CudaHandle = CUmemGenericAllocationHandle;
type AllocationType = CUmemAllocationType_enum;
type AllocationHandleType = CUmemAllocationHandleType_enum;

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


macro_rules! cuda_driver_try {
    ($expr:expr) => {{
        let result = unsafe { $expr };
        if result != CUDA_SUCCESS {
            Err(DriverError(result))
        } else {
            Ok(())
        }
    }};
}



 // I decided to implement this simple FIFO pool instead of creating a storage that returns handles and create a new pool in the memory manage module because I think it fits more with your current structure.
 // The goal of the pool is to automatically manage physical memory reusability. This way when blocks are deallocated by the pool, they are just unmapped, but memory is not completely released until pool goes out of scope or you call the [`flush`] method inside the pool.
 // Think of it as a data structure owned by the [`ExpandableStorage`] that just holds the physical memory, which when working with VMM, must be separated from virtual memory (represented as [`ExpandableBlocks`] in my implementation)
struct HandlePool {
    device_id: i32,

    max_handles: u64,
    handle_size: u64,

    // Growing map of cuda handles, mapped to virtual memory addresses.
    allocated_handles: HashMap<CUdeviceptr, CudaHandle>,
    // Free FIFO queue of free handles.
    free_queue: VecDeque<CudaHandle>,
}

impl HandlePool {
    fn new(
        device_id: i32,
        handle_size: u64,
        max_size: u64,
    ) -> Self {
        // Handle size must be a multiple of the granularity.
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

        cuda_driver_try!(cuMemGetInfo_v2(&mut free_bytes, &mut total_bytes))?;
        Ok((free_bytes as u64, total_bytes as u64))
    }

    /// Create a physical memory handle
    ///
    /// This allocates actual GPU memory that can be mapped to virtual addresses
    /// Returns the position in the vector of handles at which this handle has been placed.
    fn allocate_handle(&mut self, new_addr: CUdeviceptr) -> Result<CUdeviceptr, IoError> {
        // First, we attempt to pop from the free queue in a FIFO way.
        // The first handle that was deallocated is the first that is reused.
        if let Some(free_handle) = self.free_queue.pop_back() {
            self.allocated_handles.insert(new_addr, free_handle);
            return Ok(new_addr);
        };

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
        if cuda_driver_try!(cuMemCreate(
            &mut handle,
            self.handle_size as usize,
            &prop,
            0
        )).is_err() {

            let (free_bytes, _) = Self::get_device_memory_info().unwrap();
            if free_bytes >= self.handle_size && !self.free_queue.is_empty() {

                self.evict(1)?;
                self.allocate_handle(new_addr)?;

            }else{

                return Err(IoError::BufferTooBig(self.handle_size as usize));
            }


        }

        if cuda_driver_try!(cuMemMap(new_addr, self.handle_size as usize, 0, handle, 0)).is_ok(){


            // Set the access permissions for this device.
            self.set_access_permissions(new_addr, self.device_id);
            self.allocated_handles.insert(new_addr, handle);
            Ok(new_addr)
        }
        else{
            Err(IoError::InvalidHandle)
        }
    }



    // This are just utilities for memory management.
    fn is_full(&self, required_size: u64) -> bool{
        let (free_bytes, _) = Self::get_device_memory_info().unwrap();
        free_bytes >= required_size
    }

    fn can_evict(&self) -> bool {
        !self.free_queue.is_empty()
    }




    // Deallocation functions like dealloc and flush do not return errors in CubeCL.
    // I keep it as is for compatibility.
    pub fn deallocate_handle(&mut self, addr: CUdeviceptr)  {
        if let Some(handle) = self.allocated_handles.remove(&addr) {
            // Note: I am not sure if it is a good idea or not to let return an error here in case of failure on the device side. It is what PyTorch does (See the macro C10_CUDA_DRIVER_CHECK  at https://github.dev/pytorch/pytorch/blob/main/c10/cuda/CUDACachingAllocator.cpp)
            let _ = cuda_driver_try!(cuMemUnmap(addr, self.handle_size as usize));
            self.free_queue.push_back(handle);
        }

    }

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

    // Safely drop the first n handles from the free queue.
    fn evict(&mut self, n_handles: usize) -> Result<(), IoError> {
        self.free_queue
            .drain((self.free_queue.len().saturating_sub(n_handles))..)
            .for_each(|handle| {
                let _ = cuda_driver_try!(cuMemRelease(handle));
            });

        Ok(())
    }

    fn flush(&mut self) {
        self.free_queue.drain(..).for_each(|handle| {
            let _ = cuda_driver_try!(cuMemRelease(handle));
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

        let _ = cuda_driver_try!(cuMemSetAccess(
            addr,
            self.handle_size as usize,
            &access_desc,
            1
        ));


    }
}

impl Drop for HandlePool {
    // prevent gpu memory leaks.
    fn drop(&mut self) {
        self.allocated_handles.drain().for_each(|(_, handle)| {
            let _ = cuda_driver_try!(cuMemRelease(handle));
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



#[derive(Debug)]
pub struct ExpandableBlock {
    state: BlockState,  // Unmapped -> Mapped -> Allocated.
    num_handles: usize, // Size of the block.
    virtual_size: u64,
    base_addr: CUdeviceptr,
}

impl ExpandableBlock {

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

    fn set_unmapped(&mut self)  {
        self.state = BlockState::Unmapped;

    }

    pub fn len(&self) -> usize {
        self.num_handles
    }



    pub fn get_ptr(&self) -> CUdeviceptr {
        self.base_addr
    }
}


/// The idea of [`Expandable Storage`] is to reserve a large
/// virtual address (VA) range once and *expand* it on demand by mapping
/// additional physical pages into that range. This avoids repeatedly
/// allocating slightly-larger chunks as shapes fluctuate (e.g. batched
/// inference), which would otherwise create many small, unrecoverable
/// fragments ("slivers").
pub struct ExpandableStorage {
    device_id: i32,
    /// CUDA stream for async operations
    stream: CUstream,
    base_addr: CUdeviceptr,
    virtual_size: u64,
    handle_size: u64,
    mem_alignment: usize,
    pool: HandlePool,
    memory: HashMap<StorageId, ExpandableBlock>, // All blocks go here

    // Blocks marked for deallocation
    deallocations: Vec<StorageId>,

    ptr_bindings: PtrBindings,
}

impl ExpandableStorage {
    pub fn new(
        device_id: i32,
        stream: CUstream,
        virtual_size: u64,
        alignment: u64,
        handle_size: u64
    ) -> Self {

        let mut base_addr = 0;
        let handle_size = handle_size.next_multiple_of(alignment);
        let virtual_size = virtual_size.next_multiple_of(alignment);


        if let Err(e) = cuda_driver_try!(cuMemAddressReserve(
            &mut base_addr,
            virtual_size as usize,
            0,
            0,
            0
        )){
            panic!("Error while attempting to reserve memory: {e}");
        };


        Self {
            device_id,
            stream,
            base_addr,
            virtual_size,
            handle_size,
            mem_alignment: handle_size as usize,
            // Internal pool of handles.
            pool: HandlePool::new(
                device_id,
                handle_size,
                virtual_size
            ),
            memory: HashMap::new(),    // All blocks go here
            deallocations: Vec::new(),
            ptr_bindings: PtrBindings::new(None),
        }
    }


    // Returns the next memory address that can be assigned, to ensure that blocks are contiguous in memory.
    fn next_addr(&self) -> CUdeviceptr {
    if self.memory.is_empty() {
        self.base_addr
    } else {
        self.base_addr + (self.memory.len() as u64 * self.handle_size)
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

    #[cfg(test)]
    pub fn get_block(&self, id: &StorageId) -> Option<&ExpandableBlock> {
        self.memory.get(id)
    }

}



impl ComputeStorage for ExpandableStorage {
    type Resource = CudaResource;

    fn alignment(&self) -> usize {
        self.mem_alignment
    }


    // Exactly the same as cuda storage.
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

    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        let id = StorageId::new();

        let next_addr = self.next_addr();
        let mut block = ExpandableBlock::from_reserved(next_addr, size, self.handle_size);

        self.pool.reserve(block.num_handles)?; // Expand the size of the pool first to avoid copying the hashmap at each iteration
        // Allocate handles for this block.
        for i in 0..block.num_handles {
            let addr = block.base_addr + i as u64 * self.handle_size;
            self.pool.allocate_handle(addr)?;
        }

        block.set_mapped(); // All handles are mapped now so we set the block to mapped state
        self.memory.insert(id, block);


        Ok(StorageHandle::new(
            id,
            StorageUtilization { offset: 0, size },
        ))
    }


   // For compatibility with CubeCL current structure I keep this lazy deallocation pattern here but I think it is just not necessary, as when working with virtual memory, deallocating is just unmapping the handles of a block, which would return to the handle pool for reuse on other blocks.
   // See this blog post by NVIDIA where they show benchmarks: https://developer.nvidia.com/blog/introducing-low-level-gpu-virtual-memory-management/
   // I leave it this way for you to decide wether you want to keep it or not in the future.
    fn dealloc(&mut self, id: StorageId) {
        self.deallocations.push(id);
    }


    // Deallocates all pending handles in the
    fn flush(&mut self) {
        for id in self.deallocations.drain(..) {
            if let Some(mut block) = self.memory.remove(&id) {
                for i in 0..block.num_handles {
                        let addr = block.base_addr + i as u64 * self.handle_size;
                        self.pool.deallocate_handle(addr);
                }

            }

        }
    }
}



impl Drop for ExpandableStorage {

    fn drop(&mut self) {

        // We need to drop the pool first to prevent memory leaks.
        self.pool.allocated_handles.drain().for_each(|(_, handle)| {
            let _ = cuda_driver_try!(cuMemRelease(handle));
        });
        self.pool.allocated_handles.clear();
        self.pool.flush();
        let _ = cuda_driver_try!(cuMemAddressFree(self.base_addr, self.virtual_size as usize));
    }
}

unsafe impl Send for ExpandableStorage {}


// This enum is just to enable storage dispatch at the runtime level
pub enum CudaStorageType {
    Regular(CudaStorage),
    Expandable(ExpandableStorage),
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
