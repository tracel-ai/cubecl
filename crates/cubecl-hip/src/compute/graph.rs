use cubecl_hip_sys::hipGraphExec_t;
use cubecl_runtime::memory_management::ManagedMemoryHandle;

/// An instantiated HIP executable graph (`hipGraphExec_t`), destroyed on drop.
///
/// Owned by the [`HipServer`](super::server::HipServer) registry and referenced
/// by [`GraphId`](cubecl_runtime::id::GraphId); the raw handle never leaves the
/// server actor, which serializes access, so it is only ever touched on the one
/// thread allowed to. The client references the graph by id and, on the last
/// drop, asks the actor to release it — `graph_destroy` syncs the stream first
/// so the executable is never destroyed while a replay is still running.
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

impl Drop for HipGraph {
    fn drop(&mut self) {
        // SAFETY: `exec` was produced by `hipGraphInstantiate` and is destroyed
        // exactly once here.
        unsafe {
            cubecl_hip_sys::hipGraphExecDestroy(self.exec);
        }
    }
}
