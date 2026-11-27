use crate::compute::events::Cuda;
use crate::compute::uninit_vec;
use cubecl_core::server::IoError;
use cubecl_environment::backtrace::BackTrace;
use cubecl_server::storage::{
    HostImporter, HostRange, StorageAllocator, StorageHandle, StorageId, StorageUtilization,
};
use cudarc::driver::DriverError;
use std::collections::HashMap;

/// Whether an allocation has backing memory to free.
#[derive(Debug, Clone, Copy)]
enum AllocationKind {
    Sync,
    /// A zero-size buffer has no backing allocation.
    Empty,
}

/// Buffer storage for NVIDIA GPUs.
///
/// This struct manages memory resources for CUDA kernels, allowing them to be used as bindings
/// for launching kernels.
pub struct GpuStorage {
    memory: HashMap<StorageId, (cudarc::driver::sys::CUdeviceptr, AllocationKind, u64)>,
    deallocations: Vec<StorageId>,
    /// The bytes `memory` holds.
    allocated: u64,
    ptr_bindings: PtrBindings,
    host_imports: HostImporter<Cuda>,
}

/// A GPU memory resource allocated for CUDA using [`GpuStorage`].
#[derive(Debug)]
pub struct GpuResource {
    /// The GPU memory pointer.
    pub ptr: u64,
    /// The CUDA binding pointer.
    pub binding: *mut std::ffi::c_void,
    /// The size of the resource.
    pub size: u64,
}

impl GpuResource {
    /// Creates a new [`GpuResource`].
    pub fn new(ptr: u64, binding: *mut std::ffi::c_void, size: u64) -> Self {
        Self { ptr, binding, size }
    }
}

impl GpuStorage {
    /// Storage whose imports `stream`'s work reads. `stream` must outlive the storage.
    pub(crate) fn new(stream: cudarc::driver::sys::CUstream) -> Self {
        Self {
            memory: HashMap::new(),
            deallocations: Vec::new(),
            allocated: 0,
            ptr_bindings: PtrBindings::new(),
            host_imports: HostImporter::new(stream),
        }
    }

    /// Deallocates buffers marked for deallocation.
    ///
    /// This method processes all pending deallocations by freeing the associated GPU memory.
    fn perform_deallocations(&mut self) {
        self.host_imports.release_after_stream(&self.deallocations);
        self.deallocations
            .drain(..)
            .filter_map(|id| self.memory.remove(&id))
            // SAFETY: Nonempty pointers are owned by this storage and have not
            // been freed. Empty entries make no driver call.
            .for_each(|(ptr, kind, size)| unsafe {
                self.allocated -= size;
                let result = match kind {
                    AllocationKind::Empty => Ok(()),
                    AllocationKind::Sync => cudarc::driver::result::free_sync(ptr),
                };
                if let Err(e) = result {
                    eprintln!("CUDA free error: {}", e);
                }
            });
    }
}

// SAFETY: `GpuResource` contains CUDA device pointers that are safe to send between
// threads as long as proper stream synchronization is maintained by the caller.
unsafe impl Send for GpuResource {}
// SAFETY: `GpuStorage` is only accessed from one thread at a time via the `DeviceHandle`,
// which serializes all server access. The raw CUDA pointers it contains are never shared
// across threads without synchronization.
unsafe impl Send for GpuStorage {}

impl core::fmt::Debug for GpuStorage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GpuStorage").finish()
    }
}

/// Manages active CUDA buffer bindings in a ring buffer.
///
/// This ensures that pointers remain valid during kernel execution, preventing use-after-free errors.
struct PtrBindings {
    slots: Vec<cudarc::driver::sys::CUdeviceptr>,
    cursor: usize,
}

impl PtrBindings {
    /// Creates a new [`PtrBindings`] instance with a fixed-size ring buffer.
    fn new() -> Self {
        Self {
            // SAFETY: `CUdeviceptr` is a `u64`, valid for any bit pattern. All slots are
            // written via `register()` before being read, so uninitialized values are never observed.
            slots: unsafe { uninit_vec(crate::device::CUDA_MAX_BINDINGS as usize) },
            cursor: 0,
        }
    }

    /// Registers a new pointer in the ring buffer.
    ///
    /// # Arguments
    ///
    /// * `ptr` - The CUDA device pointer to register.
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
            Some((ptr, _, _)) => *ptr,
            None => self
                .host_imports
                .device_ptr(&handle.id)
                .expect("the handle should name a live allocation of this storage"),
        };

        let offset = handle.offset();
        let size = handle.size();
        let ptr = self.ptr_bindings.register(ptr + offset);

        GpuResource::new(
            *ptr,
            ptr as *const cudarc::driver::sys::CUdeviceptr as *mut std::ffi::c_void,
            size,
        )
    }

    fn allocate_bytes(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        let id = StorageId::new();
        // CubeCL pools these allocations itself, so the synchronous allocator is
        // the only one: CUDA's stream-ordered pool would be a second pool under
        // it, and its memory is not reachable from peer devices without
        // `cuMemPoolSetAccess`.
        let allocation = if size == 0 {
            // cuMemAlloc rejects zero bytes. Empty buffers still need a storage
            // entry for binding resolution, but no CUDA allocation or free.
            Ok((0, AllocationKind::Empty))
        } else {
            // SAFETY: The context is valid. Successful allocations remain owned
            // by `self.memory` and are freed by `perform_deallocations`.
            unsafe {
                cudarc::driver::result::malloc_sync(size as usize)
                    .map(|ptr| (ptr, AllocationKind::Sync))
            }
        };
        let (ptr, kind) = match allocation {
            Ok(allocation) => allocation,
            // Not `BufferTooBig`: that variant means the allocation can
            // never fit, and `Command::reserve` skips its reclaim-and-retry
            // when it sees it. A full device is a moment, not a verdict.
            Err(DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY)) => {
                return Err(IoError::OutOfMemory {
                    size,
                    backtrace: BackTrace::capture(),
                });
            }
            Err(other) => {
                return Err(IoError::Unknown {
                    description: format!("CUDA allocation error: {other}"),
                    backtrace: BackTrace::capture(),
                });
            }
        };

        self.memory.insert(id, (ptr, kind, size));
        self.allocated += size;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_size_allocations() {
        let context = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = context.new_stream().unwrap();

        let mut storage = GpuStorage::new(stream.cu_stream());
        for size in [0, 16] {
            let handle = storage.allocate_bytes(size).unwrap();
            let resource = storage.get(&handle);
            assert_eq!(handle.size(), size);
            assert_eq!(resource.size, size);
            assert_eq!(resource.ptr == 0, size == 0);
            assert_eq!(storage.bytes_allocated(), size);

            storage.deallocate(handle.id);
            storage.flush();
            stream.synchronize().unwrap();
            assert_eq!(storage.bytes_allocated(), 0);
        }
    }
}
