use super::HostMemoryApi;
use super::host_pinner::HostPinner;
use crate::driver::DriverError;
use cubecl_environment::bytes::Bytes;

/// The host memory a runtime pinned for its devices, each memory with how many imports of it the
/// devices hold, since a driver pins a range once per process. A runtime keeps one in a `static`,
/// which [`HostMemoryApi::host_pins`] answers with.
///
/// # Guarantees
///
/// - A memory is pinned on its first import and unpinned on its last release, once per runtime
///   whatever the number of devices importing it.
/// - Memory the driver fails to unpin stays counted as pinned, so a later import of it pins
///   nothing the driver still holds.
/// - Memory that overlaps a pin without being it pins on its own, which the driver refuses.
pub struct HostPins {
    pinner: HostPinner,
}

impl HostPins {
    /// No memory pinned.
    pub const fn new() -> Self {
        Self {
            pinner: HostPinner::new(),
        }
    }
}

impl HostPins {
    /// Counts an import of `memory`, pinning it on its first.
    pub(crate) fn pin<A: HostMemoryApi>(&self, memory: &Bytes) -> Result<(), DriverError> {
        self.pinner.pin::<A>(memory)
    }

    /// Counts a release of `memory`, unpinning it on its last.
    pub(crate) fn unpin<A: HostMemoryApi>(&self, memory: &Bytes) -> Result<(), DriverError> {
        self.pinner.unpin::<A>(memory)
    }
}

impl Default for HostPins {
    fn default() -> Self {
        Self::new()
    }
}
