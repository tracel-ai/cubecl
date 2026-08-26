use cubecl_core::server::BufferBinding;
use cubecl_runtime::memory_management::ManagedMemoryHandle;
use cudarc::driver::sys::CUgraphExec;

/// An instantiated CUDA executable graph (`CUgraphExec`), destroyed on drop.
///
/// Owned by the [`CudaServer`](super::server::CudaServer) registry and referenced
/// by [`GraphId`](cubecl_runtime::id::GraphId); the raw handle never leaves the
/// server actor, which serializes access, so it is only ever touched on the one
/// thread allowed to. The client references the graph by id and, on the last
/// drop, asks the actor to release it — `graph_destroy` syncs the stream first
/// so the executable is never destroyed while a replay is still running.
#[derive(Debug)]
pub struct CudaGraph {
    pub(crate) exec: CUgraphExec,
    /// Every buffer the captured graph touches, pinned for the graph's
    /// lifetime. A replay re-runs the recorded kernels against these exact
    /// device pointers; retaining the handles keeps the memory pool from
    /// reusing those slices (a reuse would let a later allocation share memory
    /// the replay overwrites). Dropped with the graph, releasing the memory.
    pub(crate) _retained: Vec<ManagedMemoryHandle>,
    /// The buffers the recorded launches write, deduplicated. A replay that
    /// fails to enqueue runs none of those launches, so it leaves every one of
    /// these as it was — tainting them is what makes a later read of one fail,
    /// whichever stream asks.
    pub(crate) written: Vec<BufferBinding>,
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        // SAFETY: `exec` was produced by `cuGraphInstantiateWithFlags` and is
        // destroyed exactly once here.
        unsafe {
            cudarc::driver::sys::cuGraphExecDestroy(self.exec);
        }
    }
}
