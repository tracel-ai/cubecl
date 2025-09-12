use cubecl_core::server::IoError;
use cubecl_hip_sys::HIP_SUCCESS;
use cubecl_runtime::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use std::{collections::HashMap, ffi::c_void};

/// Memory alignment for pinned host memory, set to the size of `u128` for optimal performance.
pub const PINNED_MEMORY_ALIGNMENT: usize = core::mem::size_of::<u128>();

/// Manages pinned host memory for HIP operations.
///
/// This storage handles allocation and deallocation of pinned (page-locked) host memory,
/// which is optimized for fast data transfers between host and GPU in HIP applications.
pub struct PinnedMemoryStorage {
    memory: HashMap<StorageId, PinnedMemory>,
    mem_alignment: usize,
}

/// A pinned memory resource allocated on the host.
#[derive(Debug)]
pub struct PinnedMemoryResource {
    /// Pointer to the pinned memory buffer.
    pub ptr: *mut u8,
    /// Size of the memory resource in bytes.
    pub size: usize,
}

/// Internal representation of pinned memory with associated pointers.
#[derive(Debug)]
struct PinnedMemory {
    /// Pointer to the pinned memory buffer.
    ptr: *mut c_void,
    /// Pointer-to-pointer for HIP allocation, kept alive for async operations.
    #[allow(unused)]
    ptr2ptr: *mut *mut c_void,
}

impl PinnedMemoryStorage {
    /// Creates a new [PinnedMemoryStorage] instance.
    ///
    /// Initializes the storage with the default pinned memory alignment
    /// defined by [PINNED_MEMORY_ALIGNMENT].
    pub fn new() -> Self {
        Self {
            memory: HashMap::new(),
            mem_alignment: PINNED_MEMORY_ALIGNMENT,
        }
    }
}

unsafe impl Send for PinnedMemoryResource {}
unsafe impl Send for PinnedMemoryStorage {}

impl ComputeStorage for PinnedMemoryStorage {
    type Resource = PinnedMemoryResource;

    fn alignment(&self) -> usize {
        self.mem_alignment
    }

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let memory = self
            .memory
            .get(&handle.id)
            .expect("Storage handle not found");

        let offset = handle.offset() as usize;
        let size = handle.size() as usize;

        unsafe {
            PinnedMemoryResource {
                ptr: memory.ptr.cast::<u8>().add(offset),
                size,
            }
        }
    }

    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        let resource = unsafe {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            let ptr2ptr: *mut *mut c_void = &mut ptr;

            let result = cubecl_hip_sys::hipMallocHost(ptr2ptr, size as usize);

            if result != HIP_SUCCESS {
                return Err(IoError::Unknown(format!(
                    "cuMemAllocHost_v2 failed with error code: {:?}",
                    result
                )));
            }

            PinnedMemory { ptr, ptr2ptr }
        };

        let id = StorageId::new();
        self.memory.insert(id, resource);
        Ok(StorageHandle::new(
            id,
            StorageUtilization { offset: 0, size },
        ))
    }

    fn dealloc(&mut self, id: StorageId) {
        if let Some(resource) = self.memory.remove(&id) {
            unsafe {
                cubecl_hip_sys::hipFreeHost(resource.ptr);
            }
        }
    }
}
