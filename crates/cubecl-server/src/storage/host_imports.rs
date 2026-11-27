use super::host_importer::HostImporter;
use super::{HostMemoryApi, HostRange, StorageHandle, StorageId};
use crate::server::IoError;

/// The host memory one device imported, which its kernels read in place, for a storage to hold
/// beside its device allocations.
///
/// # Assumptions
///
/// - The current thread's device is the one these imports are of whenever they import, release
///   or drop: the device's address of an import and the wait at drop are the current device's.
/// - No stream of the device records a graph while an import or a release runs: letting go of
///   an import may unpin its memory, and a driver such as HIP fails a recording on the thread
///   that pins or unpins.
///
/// # Guarantees
///
/// - Memory stays pinned and alive until the work queued on the stream before its release
///   completes, and is pinned once per runtime however many imports of it the devices hold, as
///   [`HostPins`](super::HostPins) counts them.
/// - Memory whose reads a driver failure left unknown is leaked, pinned, unless the failure
///   poisoned the device, whose work stopped with it.
/// - Dropped, the imports wait for the whole device, then let go of every import.
pub struct HostImports<A: HostMemoryApi> {
    importer: HostImporter<A>,
}

impl<A: HostMemoryApi> HostImports<A> {
    /// No imports, released behind the work of `stream`.
    ///
    /// # Safety
    ///
    /// `stream` outlives the imports: every release records an event on it.
    pub unsafe fn on_stream(stream: A::Stream) -> Self {
        Self {
            // SAFETY: the caller keeps `stream` alive for as long as the imports.
            importer: unsafe { HostImporter::on_stream(stream) },
        }
    }

    /// Imports `host`, which the device reads until the handle's id is
    /// [released](Self::release_after_stream).
    ///
    /// # Errors
    ///
    /// [`IoError::HostMemoryRefused`] when the driver refuses to pin the memory or to map it for
    /// this device, and the driver's error when it poisons the device.
    pub fn import(&mut self, host: &HostRange) -> Result<StorageHandle, IoError> {
        self.importer.import(host)
    }

    /// The device's address of import `id`, or `None` when it is no import.
    pub fn device_ptr(&self, id: &StorageId) -> Option<u64> {
        self.importer.device_ptr(id)
    }

    /// Releases the imports among `ids` behind the work queued on the stream so far, and lets go
    /// of those whose release completed.
    pub fn release_after_stream(&mut self, ids: &[StorageId]) {
        self.importer.release_after_stream(ids);
    }
}
