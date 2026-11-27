use crate::compute::status::checked;
use cubecl_common::device_handle::ExitGuard;
use cubecl_core::server::IoError;
use cubecl_environment::backtrace::BackTrace;
use cubecl_server::driver::DriverError;
use cubecl_server::storage::{HostMemory, HostReleaser, StorageId};
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

/// Host memory a device reads in place, kept alive until the work queued before its release is
/// done.
pub(super) struct HostImporter {
    /// Keyed by storage id.
    imported: HashMap<StorageId, (cubecl_hip_sys::hipDeviceptr_t, Arc<HostMemory>)>,
    stream: cubecl_hip_sys::hipStream_t,
    /// Released imports the stream may still read, held until the work queued before their release
    /// completes. Arrival: in stream order, so they complete in order and drop from the front.
    pending_releases: Vec<PendingRelease>,
}

impl HostImporter {
    pub(super) fn new(stream: cubecl_hip_sys::hipStream_t) -> Self {
        Self {
            imported: HashMap::new(),
            stream,
            pending_releases: Vec::new(),
        }
    }

    /// Imports `range` of `memory`, which the device reads until its id is released.
    pub(super) fn import(
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

    pub(super) fn device_ptr(&self, id: &StorageId) -> Option<cubecl_hip_sys::hipDeviceptr_t> {
        self.imported.get(id).map(|(ptr, _)| *ptr)
    }

    /// Releases the imports among `ids` behind the work queued on the stream so far.
    pub(super) fn release_after_stream(&mut self, ids: &[StorageId]) {
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
    /// None for no imports. Lets go of `imported` as [`let_go_unread`] says when no event can be
    /// recorded.
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
            let_go_unread(imported, &err);
            return None;
        }
        let mut release = Self { event, imported };
        // SAFETY: the event was just created, and `stream` lives as long as this storage.
        let status = unsafe { cubecl_hip_sys::hipEventRecord(event, stream) };
        release.let_go_on_error(checked("hipEventRecord", status));
        Some(release)
    }

    /// Whether the device may still read the imports. A failed query lets go of them as
    /// [`let_go_unread`] says.
    fn is_pending(&mut self) -> bool {
        // SAFETY: the event lives until this release is dropped.
        let status = unsafe { cubecl_hip_sys::hipEventQuery(self.event) };
        if status == cubecl_hip_sys::hipError_t_hipErrorNotReady {
            return true;
        }
        self.let_go_on_error(checked("hipEventQuery", status));
        false
    }

    fn let_go_on_error(&mut self, waited: Result<(), DriverError>) {
        if let Err(err) = waited {
            let_go_unread(self.imported.drain(..), &err);
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
            let_go_unread(imported, &err);
        }
    }
}

/// `imported`, whose reads `failed` left unknown, dropped where the failure poisoned the device,
/// whose work stopped with it, and leaked otherwise: a kernel may still be reading them.
fn let_go_unread(imported: impl IntoIterator<Item = Arc<HostMemory>>, failed: &DriverError) {
    if !failed.is_device_poisoned() {
        log::warn!("leaking imported host memory a kernel may still read: {failed}");
        imported.into_iter().for_each(core::mem::forget);
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

// SAFETY: the event is a driver handle, used only by the storage that holds it.
unsafe impl Send for PendingRelease {}
