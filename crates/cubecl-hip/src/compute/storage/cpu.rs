use cubecl_common::backtrace::BackTrace;
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
    stream: cubecl_hip_sys::hipStream_t,
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
    /// Device pointer: Pointer-to-pointer for HIP allocation, kept alive for async operations.
    #[allow(unused)]
    dev_ptr: *mut *mut c_void,
}

impl PinnedMemoryStorage {
    /// Creates a new [`PinnedMemoryStorage`] instance.
    ///
    /// Initializes the storage with the default pinned memory alignment
    /// defined by [`PINNED_MEMORY_ALIGNMENT`].
    pub fn new(stream: cubecl_hip_sys::hipStream_t) -> Self {
        Self {
            memory: HashMap::new(),
            mem_alignment: PINNED_MEMORY_ALIGNMENT,
            stream,
        }
    }
}

// SAFETY: `PinnedMemoryResource` contains a raw pointer to page-locked host memory.
// It is safe to send between threads because the memory remains valid and pinned
// regardless of which thread accesses it, and access is serialized by the `DeviceHandle`.
unsafe impl Send for PinnedMemoryResource {}
// SAFETY: `PinnedMemoryStorage` is only accessed from one thread at a time via the
// `DeviceHandle`, which serializes all server access. The HIP stream and pinned memory
// it manages are never shared across threads without synchronization.
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

        // SAFETY: `memory.ptr` was allocated by `hipHostMalloc` with at least `offset + size`
        // bytes. The `add(offset)` produces a pointer within the allocation bounds as
        // guaranteed by the storage handle's offset/size validation.
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
        // SAFETY: Calling HIP FFI to allocate page-locked (pinned) host memory. The
        // `hipHostMallocMapped` flag makes the memory accessible from both host and device.
        // We synchronize the stream afterward to ensure the allocation is visible.
        // The returned pointer is stored and freed via `hipFreeHost` on deallocation.
        let resource = unsafe {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            let dev_ptr: *mut *mut c_void = &mut ptr;

            let result = cubecl_hip_sys::hipHostMalloc(
                dev_ptr,
                size as usize,
                cubecl_hip_sys::hipHostMallocMapped,
            );

            if result != HIP_SUCCESS {
                return Err(IoError::Unknown {
                    description: format!("cuMemAllocHost_v2 failed with error code: {result:?}"),
                    backtrace: BackTrace::capture(),
                });
            }

            // For safety, reducing the odds of missing mapped memory page.
            cubecl_hip_sys::hipStreamSynchronize(self.stream);

            PinnedMemory { ptr, dev_ptr }
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
            // SAFETY: `resource.ptr` was allocated by `hipHostMalloc` and has not been freed
            // yet. After this call, the pointer is invalid and removed from `self.memory`.
            unsafe {
                cubecl_hip_sys::hipFreeHost(resource.ptr);
            }
        }
    }

    fn flush(&mut self) {
        // We don't wait for dealloc.
    }
}
