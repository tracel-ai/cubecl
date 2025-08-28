use super::uninit_vec;
use crate::compute::ExpandableSegment;
use cubecl_core::server::IoError;
use cubecl_runtime::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use cudarc::driver::{DriverError, sys::CUstream};
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

/// CUDA storage version with VMM support.
/// Handles VMM based allocations (see vmm.rs, specifically the [`ExpandableSegments`] struct)
pub struct ExpandableCudaStorage {
    /// Device ID for this storage
    device_id: i32,
    /// CUDA stream for async operations
    stream: CUstream,
    /// Expandable segment for VMM allocations
    expandable_segment: ExpandableSegment,

    /// Memory alignment requirement
    mem_alignment: usize,

    /// Pointer bindings for kernel parameters (from original CudaStorage)
    /// This circular buffer manages device pointers for kernel launches
    ptr_bindings: PtrBindings,
}

/// Implement ComputeStorage trait for VMM-enabled storage
impl ComputeStorage for ExpandableCudaStorage {
    type Resource = CudaResource;

    /// Return memory alignment requirement
    fn alignment(&self) -> usize {
        self.mem_alignment
    }

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let mapping = self
            .expandable_segment
            .get_mapping(handle.id) // This function returns a mapped memory region
            .expect("Storage handle not found");

        let base_ptr = mapping.device_ptr();
        let final_ptr = base_ptr + handle.offset();
        let binding_ptr = self.ptr_bindings.register(final_ptr);

        CudaResource::new(
            final_ptr,
            binding_ptr as *const u64 as *mut std::ffi::c_void,
            handle.offset(),
            handle.size(),
        )
    }

    /// Allocates memory
    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        let id = StorageId::new();
        let alloc = self.expandable_segment.alloc_and_map(size, id)?; // Allocates a new mapped region

        Ok(StorageHandle::new(
            id,
            StorageUtilization {
                offset: 0,
                size: alloc.size,
            },
        ))
    }

    /// Deallocate memory directly from the storage, but it is kept in the mapped vector inside the [`ExpandableSegment`].
    fn dealloc(&mut self, id: StorageId) {
        // self.deallocations.push(id);

        self.expandable_segment
            .dealloc(id)
            .unwrap_or_else(|e| panic!("Failed to dealloc id {:?}: {}", id, e));
    }

    /// Unmaps all mapped blocks
    fn flush(&mut self) {
        self.expandable_segment
            .release_all(true)
            .unwrap_or_else(|e| panic!("Failed to flush VMM storage: {}", e)); // Unmaps all blocks and releases the physical memory.
        self.expandable_segment.clear_handles();
    }
}

unsafe impl Send for ExpandableCudaStorage {}

impl ExpandableCudaStorage {
    /// Create new VMM-enabled CUDA storage
    ///
    /// # Arguments
    /// - `device_id`: CUDA device index
    /// - `stream`: CUDA stream for operations
    /// - `mem_alignment`: Memory alignment requirement
    /// - `max_bindings`: Size of pointer binding circular buffer
    pub fn new(
        device_id: i32,
        stream: CUstream,
        mem_alignment: usize,
        virtual_size: u64,
        handle_size: u64,
        max_bindings: usize,
        gc_threshold_ratio: f64,
    ) -> Self {
        let expandable_segment =
            ExpandableSegment::new(device_id, virtual_size, handle_size, gc_threshold_ratio)
                .expect("Failed to create ExpandableSegment");

        Self {
            device_id,
            stream,
            expandable_segment,
            mem_alignment,
            ptr_bindings: PtrBindings::new(Some(max_bindings)),
        }
    }
}

/// Unified storage type that supports both regular and VMM allocation
///
/// This enum allows the CUDA runtime to use either storage backend transparently.
/// The choice is made at initialization time based on RuntimeOptions.
pub enum CudaStorageType {
    /// Regular CUDA storage using cudaMalloc/cudaFree
    Regular(CudaStorage),
    /// VMM-enabled CUDA storage with expandable segments
    Vmm(ExpandableCudaStorage),
}

impl CudaStorageType {
    /// Create regular CUDA storage
    pub fn regular(mem_alignment: usize, stream: CUstream) -> Self {
        Self::Regular(CudaStorage::new(mem_alignment, stream))
    }

    /// Create VMM-enabled CUDA storage with custom configuration
    pub fn vmm(
        device_id: i32,
        stream: CUstream,
        mem_alignment: usize,
        max_bindings: usize,
        virtual_size: u64,
        handle_size: u64,
        gc_threshold_ratio: f64,
    ) -> Self {
        let vmm = ExpandableCudaStorage::new(
            device_id,
            stream,
            mem_alignment,
            virtual_size,
            handle_size,
            max_bindings,
            gc_threshold_ratio,
        );
        Self::Vmm(vmm)
    }

    /// Check if this storage uses VMM
    pub fn uses_vmm(&self) -> bool {
        matches!(self, Self::Vmm(_))
    }

    /// Get a string description of the storage type for logging
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Regular(_) => "Regular",
            Self::Vmm(_) => "VMM",
        }
    }
}

/// Implement ComputeStorage by delegating to the appropriate backend
///
/// This ensures that both storage types have identical interfaces and can be
/// used interchangeably by the memory management system.
impl ComputeStorage for CudaStorageType {
    type Resource = CudaResource;

    fn alignment(&self) -> usize {
        match self {
            Self::Regular(storage) => storage.alignment(),
            Self::Vmm(storage) => storage.alignment(),
        }
    }

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        match self {
            Self::Regular(storage) => storage.get(handle),
            Self::Vmm(storage) => storage.get(handle),
        }
    }

    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        match self {
            Self::Regular(storage) => storage.alloc(size),
            Self::Vmm(storage) => storage.alloc(size),
        }
    }

    fn dealloc(&mut self, id: StorageId) {
        match self {
            Self::Regular(storage) => storage.dealloc(id),
            Self::Vmm(storage) => storage.dealloc(id),
        }
    }

    fn flush(&mut self) {
        match self {
            Self::Regular(storage) => storage.flush(),
            Self::Vmm(storage) => storage.flush(),
        }
    }
}

unsafe impl Send for CudaStorageType {}
