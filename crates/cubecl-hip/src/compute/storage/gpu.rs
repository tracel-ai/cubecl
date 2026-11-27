use super::host_import::HipPinner;
use crate::compute::status::checked;
use cubecl_core::server::IoError;
use cubecl_server::storage::{
    HostImporter, HostRange, StorageAllocator, StorageHandle, StorageId, StorageUtilization,
};
use std::collections::HashMap;

use crate::AMD_MAX_BINDINGS;

/// Buffer storage for AMD GPUs.
///
/// This struct manages memory resources for HIP kernels, allowing them to be used as bindings
/// for launching kernels.
pub struct GpuStorage {
    memory: HashMap<StorageId, (cubecl_hip_sys::hipDeviceptr_t, u64)>,
    /// The bytes `memory` holds.
    allocated: u64,
    deallocations: Vec<StorageId>,
    ptr_bindings: PtrBindings,
    host_imports: HostImporter<HipPinner>,
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
    /// Deallocates buffers marked for deallocation.
    ///
    /// This method processes all pending deallocations by freeing the associated GPU memory.
    pub fn perform_deallocations(&mut self) {
        self.host_imports.release_after_stream(&self.deallocations);
        for id in self.deallocations.drain(..) {
            if let Some((ptr, size)) = self.memory.remove(&id) {
                self.allocated -= size;
                // SAFETY: `ptr` was obtained from a prior `hipMalloc` call and
                // has not been freed yet. `hipFree` synchronizes the device, so
                // in-flight work never sees the page disappear.
                let status = unsafe { cubecl_hip_sys::hipFree(ptr) };
                // Logged, not reported: this runs on a deallocation path with
                // no caller left to hand an error to.
                if let Err(err) = checked("hipFree", status) {
                    log::warn!("releasing a device page failed: {err}");
                }
            }
        }
    }
}

impl GpuStorage {
    /// Storage whose imports `stream`'s work reads. `stream` must outlive the storage.
    pub(crate) fn new(stream: cubecl_hip_sys::hipStream_t) -> Self {
        Self {
            memory: HashMap::new(),
            allocated: 0,
            deallocations: Vec::new(),
            ptr_bindings: PtrBindings::new(),
            host_imports: HostImporter::new(HipPinner::new(stream)),
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

impl StorageAllocator for GpuStorage {
    type Resource = GpuResource;

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let ptr = match self.memory.get(&handle.id) {
            Some((ptr, _)) => *ptr,
            None => self
                .host_imports
                .device_ptr(&handle.id)
                .expect("the handle should name a live allocation of this storage")
                as cubecl_hip_sys::hipDeviceptr_t,
        };
        let ptr = ptr as u64;

        let offset = handle.offset();
        let size = handle.size();
        let ptr = self.ptr_bindings.register(ptr + offset);

        GpuResource::new(
            *ptr as cubecl_hip_sys::hipDeviceptr_t,
            std::ptr::from_ref(ptr) as *mut std::ffi::c_void,
            size,
        )
    }

    fn allocate_bytes(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        let id = StorageId::new();
        // Plain (synchronous) `hipMalloc` on purpose: `hipMallocAsync` draws
        // from the driver's async mempool, which on ROCm never returns freed
        // memory to the OS (`hipFreeAsync` + `hipMemPoolTrimTo(0)` keep the
        // high-water mark forever — measured on gfx1151). The pools above
        // amortize allocations into large, rare pages, so the synchronous
        // call costs nothing where it matters — and freed pages really leave
        // the process.
        //
        // SAFETY: Calling HIP FFI to allocate device memory. The pointer is
        // stored in `self.memory` and freed via `hipFree` on deallocation.
        unsafe {
            let mut ptr: *mut ::std::os::raw::c_void = std::ptr::null_mut();
            let status = cubecl_hip_sys::hipMalloc(&mut ptr, size as usize);

            checked("hipMalloc", status)?;

            self.memory.insert(id, (ptr, size));
            self.allocated += size;
        };

        Ok(StorageHandle::new(
            id,
            StorageUtilization {
                offset_bytes: 0,
                size_bytes: size,
            },
        ))
    }

    fn import_host_memory(&mut self, host: &HostRange) -> Result<StorageHandle, IoError> {
        let id = self.host_imports.import(host)?;
        Ok(StorageHandle::new(
            id,
            StorageUtilization {
                offset_bytes: 0,
                size_bytes: host.len(),
            },
        ))
    }

    fn deallocate(&mut self, id: StorageId) {
        self.deallocations.push(id);
    }

    fn flush(&mut self) {
        self.perform_deallocations();
    }

    fn bytes_allocated(&self) -> u64 {
        self.allocated
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
