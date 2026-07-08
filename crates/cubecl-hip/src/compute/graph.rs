use cubecl_hip_sys::hipGraphExec_t;
use cubecl_runtime::memory_management::ManagedMemoryHandle;

/// An instantiated HIP executable graph (`hipGraphExec_t`), destroyed on drop.
///
/// The raw handle is only ever touched from inside the server actor, which
/// serializes access, so it is sound to move across the actor boundary.
#[derive(Debug)]
pub struct HipGraph {
    pub(crate) exec: hipGraphExec_t,
    /// Every buffer the captured graph touches, pinned for the graph's
    /// lifetime. A replay re-runs the recorded kernels against these exact
    /// device pointers; retaining the handles keeps the memory pool from
    /// reusing those slices (a reuse would let a later allocation share memory
    /// the replay overwrites). Dropped with the graph, releasing the memory.
    pub(crate) _retained: Vec<ManagedMemoryHandle>,
}

// SAFETY: the exec handle is used exclusively on the owning device's server
// actor (see the ComputeServer impl), never concurrently.
unsafe impl Send for HipGraph {}
unsafe impl Sync for HipGraph {}

impl Drop for HipGraph {
    fn drop(&mut self) {
        // SAFETY: `exec` was produced by `hipGraphInstantiate` and is destroyed
        // exactly once here.
        unsafe {
            cubecl_hip_sys::hipGraphExecDestroy(self.exec);
        }
    }
}
