use super::uninit_vec;
use cubecl_core::server::IoError;
use cubecl_runtime::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use cubecl_runtime::storage::{
    PhysicalStorageHandle, PhysicalStorageId, VirtualAddressSpaceHandle, VirtualMapping,
    VirtualSpaceId, VirtualStorage,
};
use cudarc::driver::DriverError;

use cudarc::driver::sys::{
    CUdeviceptr, CUmemAccess_flags_enum::CU_MEM_ACCESS_FLAGS_PROT_READWRITE, CUmemAccessDesc,
    CUmemAllocationHandleType_enum, CUmemAllocationProp,
    CUmemAllocationType_enum::CU_MEM_ALLOCATION_TYPE_PINNED, CUmemGenericAllocationHandle,
    CUmemLocation, CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_DEVICE, CUstream, cuMemAddressFree,
    cuMemAddressReserve, cuMemCreate, cuMemMap, cuMemRelease, cuMemSetAccess, cuMemUnmap,
    cudaError_enum::CUDA_SUCCESS,
};
use std::collections::HashMap;

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
