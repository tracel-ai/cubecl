use cubecl_common::backtrace::BackTrace;
use cubecl_core::server::IoError;
use cubecl_runtime::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use std::{collections::HashMap, ffi::c_void};

/// Memory alignment for pinned host memory, set to the size of `u128` for optimal performance.
pub const PINNED_MEMORY_ALIGNMENT: usize = core::mem::size_of::<u128>();

/// Manages pinned host memory for CUDA operations.
///
/// This storage handles allocation and deallocation of pinned (page-locked) host memory,
/// which is optimized for fast data transfers between host and GPU in CUDA applications.
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
            mem_alignment: PINNED_MEMORY_ALIGNMENT,
        }
    }
}

// SAFETY: `PinnedMemoryResource` contains a raw pointer to page-locked host memory.
// It is safe to send between threads because the memory remains valid and pinned
// regardless of which thread accesses it, and access is serialized by the `DeviceHandle`.
unsafe impl Send for PinnedMemoryResource {}
// SAFETY: `PinnedMemoryStorage` is only accessed from one thread at a time via the
// `DeviceHandle`, which serializes all server access. The pinned memory it manages
// is never shared across threads without synchronization.
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

        // SAFETY: `memory.ptr` was allocated by `cuMemAllocHost_v2` with at least
        // `offset + size` bytes. The `add(offset)` produces a pointer within the allocation
        // bounds as guaranteed by the storage handle's offset/size validation.
        unsafe {
            PinnedMemoryResource {
                ptr: memory.ptr.cast::<u8>().add(offset),
                size,
            }
        }
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, size))
    )]
    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
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

            PinnedMemory { ptr, ptr2ptr }
        };

        let id = StorageId::new();
        self.memory.insert(id, resource);
        Ok(StorageHandle::new(
            id,
            StorageUtilization { offset: 0, size },
        ))
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    fn dealloc(&mut self, id: StorageId) {
        if let Some(resource) = self.memory.remove(&id) {
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
}
