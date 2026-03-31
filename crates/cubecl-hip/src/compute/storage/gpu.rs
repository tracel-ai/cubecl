use cubecl_common::backtrace::BackTrace;
use cubecl_core::server::IoError;
use cubecl_hip_sys::HIP_SUCCESS;
use cubecl_runtime::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use std::collections::HashMap;

use crate::AMD_MAX_BINDINGS;

/// Buffer storage for AMD GPUs.
///
/// This struct manages memory resources for HIP kernels, allowing them to be used as bindings
/// for launching kernels.
pub struct GpuStorage {
    mem_alignment: usize,
    memory: HashMap<StorageId, cubecl_hip_sys::hipDeviceptr_t>,
    deallocations: Vec<StorageId>,
    ptr_bindings: PtrBindings,
    stream: cubecl_hip_sys::hipStream_t,
}

/// A GPU memory resource allocated for HIP using [`GpuStorage`].
#[derive(new, Debug)]
pub struct GpuResource {
    /// The GPU memory pointer.
    pub ptr: cubecl_hip_sys::hipDeviceptr_t,
    /// The HIP binding pointer.
    pub binding: cubecl_hip_sys::hipDeviceptr_t,
    /// The size of the resource.
    pub size: u64,
}

impl GpuStorage {
    /// Creates a new [`GpuStorage`] instance for the specified HIP stream.
    ///
    /// # Arguments
    ///
    /// * `mem_alignment` - The memory alignment requirement in bytes.
    pub fn new(mem_alignment: usize, stream: cubecl_hip_sys::hipStream_t) -> Self {
        Self {
            mem_alignment,
            memory: HashMap::new(),
            deallocations: Vec::new(),
            ptr_bindings: PtrBindings::new(),
            stream,
        }
    }

    /// Deallocates buffers marked for deallocation.
    ///
    /// This method processes all pending deallocations by freeing the associated GPU memory.
    pub fn perform_deallocations(&mut self) {
        for id in self.deallocations.drain(..) {
            if let Some(ptr) = self.memory.remove(&id) {
                // SAFETY: `ptr` was obtained from a prior `hipMallocAsync` call and has not
                // been freed yet. `self.stream` is the same stream used for allocation.
                unsafe {
                    cubecl_hip_sys::hipFreeAsync(ptr, self.stream);
                }
            }
        }
    }
}

/// Manages active HIP buffer bindings in a ring buffer.
///
/// This ensures that pointers remain valid during kernel execution, preventing use-after-free errors.
struct PtrBindings {
    slots: Vec<u64>,
    cursor: usize,
}

impl PtrBindings {
    /// Creates a new [`PtrBindings`] instance with a fixed-size ring buffer.
    fn new() -> Self {
        Self {
            slots: vec![0; AMD_MAX_BINDINGS as usize],
            cursor: 0,
        }
    }

    /// Registers a new pointer in the ring buffer.
    ///
    /// # Arguments
    ///
    /// * `ptr` - The HIP device pointer to register.
    ///
    /// # Returns
    ///
    /// A reference to the registered pointer.
    fn register(&mut self, ptr: u64) -> &u64 {
        self.slots[self.cursor] = ptr;
        let ptr_ref = self.slots.get(self.cursor).unwrap();

        self.cursor += 1;

        // Reset the cursor when the ring buffer is full.
        if self.cursor >= self.slots.len() {
            self.cursor = 0;
        }

        ptr_ref
    }
}

impl ComputeStorage for GpuStorage {
    type Resource = GpuResource;

    fn alignment(&self) -> usize {
        self.mem_alignment
    }

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let ptr = (*self.memory.get(&handle.id).unwrap()) as u64;

        let offset = handle.offset();
        let size = handle.size();
        let ptr = self.ptr_bindings.register(ptr + offset);

        GpuResource::new(
            *ptr as cubecl_hip_sys::hipDeviceptr_t,
            std::ptr::from_ref(ptr) as *mut std::ffi::c_void,
            size,
        )
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, size))
    )]
    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        let id = StorageId::new();
        // SAFETY: Calling HIP FFI to allocate device memory asynchronously. The returned
        // pointer is valid after stream synchronization (performed below). The pointer is
        // stored in `self.memory` and will be freed via `hipFreeAsync` on deallocation.
        unsafe {
            let mut ptr: *mut ::std::os::raw::c_void = std::ptr::null_mut();
            let status = cubecl_hip_sys::hipMallocAsync(&mut ptr, size as usize, self.stream);

            match status {
                HIP_SUCCESS => {}
                other => {
                    return Err(IoError::Unknown {
                        description: format!("HIP allocation error: {other}"),
                        backtrace: BackTrace::capture(),
                    });
                }
            }

            // For safety, reducing the odds of missing mapped memory page.
            cubecl_hip_sys::hipStreamSynchronize(self.stream);

            self.memory.insert(id, ptr);
        };

        Ok(StorageHandle::new(
            id,
            StorageUtilization { offset: 0, size },
        ))
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    fn dealloc(&mut self, id: StorageId) {
        self.deallocations.push(id);
    }

    fn flush(&mut self) {
        self.perform_deallocations();
    }
}

// SAFETY: `GpuStorage` is only accessed from one thread at a time via the `DeviceHandle`,
// which serializes all server access. The raw HIP pointers it contains are never shared
// across threads without synchronization.
unsafe impl Send for GpuStorage {}
// SAFETY: `GpuResource` contains raw HIP device pointers that are safe to send between
// threads as long as proper stream synchronization is maintained by the caller.
unsafe impl Send for GpuResource {}

impl core::fmt::Debug for GpuStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("GpuStorage".to_string().as_str())
    }
}
