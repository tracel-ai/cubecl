use cubecl_core::server::IoError;
use cubecl_environment::backtrace::BackTrace;
use cubecl_server::storage::{
    PinnedMemoryResource, StorageAllocator, StorageHandle, StorageId, StorageUtilization,
};
use std::{collections::HashMap, ffi::c_void};

/// Manages pinned host memory for CUDA operations.
///
/// This storage handles allocation and deallocation of pinned (page-locked) host memory,
/// which is optimized for fast data transfers between host and GPU in CUDA applications.
pub struct PinnedMemoryStorage {
    memory: HashMap<StorageId, PinnedMemory>,
    /// The bytes `memory` holds.
    allocated: u64,
}

/// Internal representation of pinned memory with associated pointers.
#[derive(Debug)]
struct PinnedMemory {
    /// Pointer to the pinned memory buffer.
    ptr: *mut c_void,
    /// The size of the buffer in bytes.
    size: u64,
    /// Pointer-to-pointer for CUDA allocation, kept alive for async operations.
    #[allow(unused)]
    ptr2ptr: *mut *mut c_void,
}

impl PinnedMemoryStorage {
    /// Creates a new [`PinnedMemoryStorage`] instance.
    ///
    /// Initializes the storage with the default pinned memory alignment
    /// defined by [`PINNED_MEMORY_ALIGNMENT`].
    pub fn new() -> Self {
        Self {
            memory: HashMap::new(),
            allocated: 0,
        }
    }
}

// SAFETY: `PinnedMemoryStorage` is only accessed from one thread at a time via the
// `DeviceHandle`, which serializes all server access. The pinned memory it manages
// is never shared across threads without synchronization.
unsafe impl Send for PinnedMemoryStorage {}

impl StorageAllocator for PinnedMemoryStorage {
    type Resource = PinnedMemoryResource;

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let memory = self
            .memory
            .get(&handle.id)
            .expect("the handle should name a live allocation of this storage");

        let offset = handle.offset();
        let size = handle.size();
        // The pointer arithmetic below is sound only inside the allocation,
        // so this is checked rather than assumed.
        assert!(
            offset + size <= memory.size,
            "the handle should lie within its allocation"
        );

        // SAFETY: `memory.ptr` was allocated by `cuMemAllocHost_v2` with `memory.size`
        // bytes, and the assert above keeps `offset + size` within them.
        unsafe {
            PinnedMemoryResource {
                ptr: memory.ptr.cast::<u8>().add(offset as usize),
                size: size as usize,
            }
        }
    }

    fn allocate_bytes(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        // SAFETY: Calling CUDA driver FFI to allocate page-locked (pinned) host memory.
        // The returned pointer is stored and freed via `cuMemFreeHost` on deallocation.
        let resource = unsafe {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            let ptr2ptr: *mut *mut c_void = &mut ptr;

            // Allocate pinned host memory using cuMemAllocHost_v2
            let result = cudarc::driver::sys::cuMemAllocHost_v2(ptr2ptr, size as usize);

            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Err(IoError::Unknown {
                    description: format!("cuMemAllocHost_v2 failed with error code: {result:?}"),
                    backtrace: BackTrace::capture(),
                });
            }

            PinnedMemory { ptr, size, ptr2ptr }
        };

        let id = StorageId::new();
        self.memory.insert(id, resource);
        self.allocated += size;
        Ok(StorageHandle::new(
            id,
            StorageUtilization {
                offset_bytes: 0,
                size_bytes: size,
            },
        ))
    }

    fn deallocate(&mut self, id: StorageId) {
        if let Some(resource) = self.memory.remove(&id) {
            self.allocated -= resource.size;
            // SAFETY: `resource.ptr` was allocated by `cuMemAllocHost_v2` and has not been
            // freed yet. After this call, the pointer is invalid and removed from `self.memory`.
            unsafe {
                cudarc::driver::sys::cuMemFreeHost(resource.ptr);
            }
        }
    }

    fn flush(&mut self) {
        // We don't wait for dealloc.
    }

    fn bytes_allocated(&self) -> u64 {
        self.allocated
    }
}
