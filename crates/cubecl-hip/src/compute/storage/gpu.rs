use crate::compute::status::checked;
use cubecl_common::device_handle::ExitGuard;
use cubecl_core::server::IoError;
use cubecl_environment::backtrace::BackTrace;
use cubecl_server::driver::DriverError;
use cubecl_server::storage::{
    HostMemory, HostReleaser, StorageAllocator, StorageHandle, StorageId, StorageUtilization,
};
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

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
    host_imports: HostImporter,
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
            host_imports: HostImporter::new(stream),
        }
    }
}

/// Host memory a device reads in place, kept alive until the work queued before its release is
/// done.
struct HostImporter {
    /// Keyed by storage id.
    imported: HashMap<StorageId, (cubecl_hip_sys::hipDeviceptr_t, Arc<HostMemory>)>,
    stream: cubecl_hip_sys::hipStream_t,
    /// Released imports the stream may still read, held until the work queued before their release
    /// completes. Arrival: in stream order, so they complete in order and drop from the front.
    pending_releases: Vec<PendingRelease>,
}

impl HostImporter {
    fn new(stream: cubecl_hip_sys::hipStream_t) -> Self {
        Self {
            imported: HashMap::new(),
            stream,
            pending_releases: Vec::new(),
        }
    }

    /// Imports `range` of `memory`, which the device reads until its id is released.
    fn import(
        &mut self,
        memory: &Arc<HostMemory>,
        range: Range<u64>,
    ) -> Result<StorageId, IoError> {
        // Imports dropped since the last flush release their pinning before this one adds more.
        self.drop_completed();
        memory.register_once(HipHostReleaser::new)?;
        let host = memory.as_ptr().wrapping_add(range.start as usize);
        let mut ptr: cubecl_hip_sys::hipDeviceptr_t = std::ptr::null_mut();
        // SAFETY: `host` lies in memory registered as mapped, so the driver has a device address
        // for it, written through `ptr`.
        let status =
            unsafe { cubecl_hip_sys::hipHostGetDevicePointer(&mut ptr, host as *mut _, 0) };
        refused_unless_poisoned("hipHostGetDevicePointer", status, memory.len())?;

        let id = StorageId::new();
        self.imported.insert(id, (ptr, memory.clone()));
        Ok(id)
    }

    fn device_ptr(&self, id: &StorageId) -> Option<cubecl_hip_sys::hipDeviceptr_t> {
        self.imported.get(id).map(|(ptr, _)| *ptr)
    }

    /// Releases the imports among `ids` behind the work queued on the stream so far.
    fn release_after_stream(&mut self, ids: &[StorageId]) {
        let imported: Vec<_> = ids
            .iter()
            .filter_map(|id| self.imported.remove(id))
            .map(|(_, memory)| memory)
            .collect();
        self.pending_releases
            .extend(PendingRelease::record(self.stream, imported));
        self.drop_completed();
    }

    /// Drops the imports whose release completed. Never while a stream records: dropping the last
    /// owner unpins the memory, and HIP fails a recording on the thread that pins or unpins.
    fn drop_completed(&mut self) {
        let done = self
            .pending_releases
            .iter_mut()
            .position(PendingRelease::is_pending)
            .unwrap_or(self.pending_releases.len());
        self.pending_releases.drain(..done);
    }
}

/// Imports deallocated on a stream, dropped once the work queued there before is done. Another
/// stream's reads keep their bindings, so the memory is not deallocated before those complete.
struct PendingRelease {
    event: cubecl_hip_sys::hipEvent_t,
    /// Arrival: dropped together once the event completes.
    imported: Vec<Arc<HostMemory>>,
}

impl PendingRelease {
    /// None for no imports. Leaks `imported` when no event can be recorded: a kernel may still be
    /// reading.
    fn record(stream: cubecl_hip_sys::hipStream_t, imported: Vec<Arc<HostMemory>>) -> Option<Self> {
        if imported.is_empty() {
            return None;
        }
        let mut event: cubecl_hip_sys::hipEvent_t = std::ptr::null_mut();
        // SAFETY: `event` is a valid out-pointer, used only once the status says it was written.
        let status = unsafe {
            cubecl_hip_sys::hipEventCreateWithFlags(
                &mut event,
                cubecl_hip_sys::hipEventDisableTiming,
            )
        };
        if let Err(err) = checked("hipEventCreateWithFlags", status) {
            log::warn!("leaking imported host memory a kernel may still read: {err}");
            core::mem::forget(imported);
            return None;
        }
        let mut release = Self { event, imported };
        // SAFETY: the event was just created, and `stream` lives as long as this storage.
        let status = unsafe { cubecl_hip_sys::hipEventRecord(event, stream) };
        release.leak_on_error(checked("hipEventRecord", status));
        Some(release)
    }

    /// Whether the device may still read the imports. A failed query leaks them.
    fn is_pending(&mut self) -> bool {
        // SAFETY: the event lives until this release is dropped.
        let status = unsafe { cubecl_hip_sys::hipEventQuery(self.event) };
        if status == cubecl_hip_sys::hipError_t_hipErrorNotReady {
            return true;
        }
        self.leak_on_error(checked("hipEventQuery", status));
        false
    }

    fn leak_on_error(&mut self, waited: Result<(), DriverError>) {
        if let Err(err) = waited {
            log::warn!("leaking imported host memory a kernel may still read: {err}");
            self.imported.drain(..).for_each(core::mem::forget);
        }
    }
}

impl Drop for PendingRelease {
    fn drop(&mut self) {
        // SAFETY: the event was created in `record` and is destroyed once, here.
        let status = unsafe { cubecl_hip_sys::hipEventDestroy(self.event) };
        if let Err(err) = checked("hipEventDestroy", status) {
            log::warn!("destroying an import's release event failed: {err}");
        }
    }
}

impl Drop for HostImporter {
    fn drop(&mut self) {
        // The whole device, not this stream: another stream that failed to wait on its reads
        // keeps their bindings, and teardown drops the imports regardless.
        let mut imported: Vec<_> = self
            .imported
            .drain()
            .map(|(_, (_, memory))| memory)
            .collect();
        imported.extend(
            self.pending_releases
                .iter_mut()
                .flat_map(|release| release.imported.drain(..)),
        );
        self.pending_releases.clear();
        if imported.is_empty() {
            return;
        }
        // SAFETY: no pointer is passed; the call waits for this device's work.
        let status = unsafe { cubecl_hip_sys::hipDeviceSynchronize() };
        if let Err(err) = checked("hipDeviceSynchronize", status) {
            log::warn!("leaking imported host memory a kernel may still read: {err}");
            core::mem::forget(imported);
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
                .expect("the handle should name a live allocation of this storage"),
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

    fn import_host_memory(
        &mut self,
        memory: &Arc<HostMemory>,
        range: Range<u64>,
    ) -> Result<StorageHandle, IoError> {
        memory.ensure_importable(&range)?;
        let id = self.host_imports.import(memory, range.clone())?;
        Ok(StorageHandle::new(
            id,
            StorageUtilization {
                offset_bytes: 0,
                size_bytes: range.end - range.start,
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

/// The pinning and mapping of a [`HostMemory`] into every device of the process, released when the
/// memory is dropped.
struct HipHostReleaser {
    ptr: *mut std::ffi::c_void,
}

impl HipHostReleaser {
    fn new(memory: &HostMemory) -> Result<Self, IoError> {
        let ptr = memory.as_ptr() as *mut std::ffi::c_void;
        let flags = cubecl_hip_sys::hipHostRegisterPortable | cubecl_hip_sys::hipHostRegisterMapped;
        // SAFETY: `ptr` is `memory.len()` bytes the host owns until this registration is dropped,
        // which `HostMemory` does before it frees them.
        let status = unsafe { cubecl_hip_sys::hipHostRegister(ptr, memory.len(), flags) };
        refused_unless_poisoned("hipHostRegister", status, memory.len())?;
        Ok(Self { ptr })
    }
}

/// A failed import as [`IoError::HostMemoryRefused`], unless the status poisons the device.
fn refused_unless_poisoned(
    op: &'static str,
    status: cubecl_hip_sys::hipError_t,
    size_bytes: usize,
) -> Result<(), IoError> {
    checked(op, status).map_err(|err| match err.is_device_poisoned() {
        true => err.into(),
        // The KFD system memory limit is what refuses a large pinning.
        false => IoError::HostMemoryRefused {
            reason: format!("importing {size_bytes} bytes of host memory: {err}").into(),
            backtrace: BackTrace::capture(),
        },
    })
}

impl HostReleaser for HipHostReleaser {
    fn release(&mut self) -> bool {
        // The last owner may drop the memory on any thread, while the process exits; the OS
        // unpins it then.
        let Some(_guard) = ExitGuard::acquire() else {
            return false;
        };
        // SAFETY: `ptr` was registered in `new` and is unregistered once, here.
        let status = unsafe { cubecl_hip_sys::hipHostUnregister(self.ptr) };
        match checked("hipHostUnregister", status) {
            Ok(()) => true,
            Err(err) => {
                log::warn!("leaking host memory HIP may still read: {err}");
                false
            }
        }
    }
}

// SAFETY: the pointer is only handed back to the driver, which accepts it from any thread.
unsafe impl Send for HipHostReleaser {}
// SAFETY: as above; nothing reads through it.
unsafe impl Sync for HipHostReleaser {}

// SAFETY: `GpuStorage` is only accessed from one thread at a time via the `DeviceHandle`,
// which serializes all server access. The raw HIP pointers it contains are never shared
// across threads without synchronization.
unsafe impl Send for GpuStorage {}
// SAFETY: the event is a driver handle, used only by the storage that holds it.
unsafe impl Send for PendingRelease {}
// SAFETY: `GpuResource` contains raw HIP device pointers that are safe to send between
// threads as long as proper stream synchronization is maintained by the caller.
unsafe impl Send for GpuResource {}

impl core::fmt::Debug for GpuStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("GpuStorage".to_string().as_str())
    }
}
