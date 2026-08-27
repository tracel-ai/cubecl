//! Graph capture: opening a window on a stream, sealing what it recorded into
//! a replayable executable, and the registry of the executables that sealed.
//!
//! The driver's rules are what shapes this. A stream in capture mode records
//! every launch issued on it, so a window that opened has to be closed even
//! when what it recorded is worthless. Nothing may allocate inside the window,
//! so the pools are warmed before it opens and pinned after it closes. And a
//! host sync would abort the recording, so the fenced flushes the execution
//! path would otherwise run are deferred across it.

use cubecl_core::server::{BufferBinding, ServerError};
use cubecl_environment::backtrace::BackTrace;
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
    /// The buffers the recorded launches write, deduplicated. A replay that
    /// fails to enqueue runs none of those launches, so it leaves every one of
    /// these as it was — tainting them is what makes a later read of one fail,
    /// whichever stream asks.
    pub(crate) written: Vec<BufferBinding>,
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

/// Turn a HIP status into a [`ServerError`], naming the failed operation.
pub(crate) fn hip_check(op: &str, status: cubecl_hip_sys::hipError_t) -> Result<(), ServerError> {
    if status == cubecl_hip_sys::HIP_SUCCESS {
        Ok(())
    } else {
        Err(ServerError::Generic {
            reason: format!("{op} failed with HIP status {status}"),
            backtrace: BackTrace::capture(),
        })
    }
}

/// Count the memory-allocation/free nodes recorded in `graph`.
///
/// A captured graph is only replayable if it owns no memory nodes: the driver refuses to
/// relaunch a graph whose allocation nodes have not been freed. Every allocation the capture
/// window needs must therefore be served by the already-warmed persistent pool — the window
/// growing the pool is precisely the condition this detects.
///
/// # Safety
///
/// `graph` must be a valid, not-yet-destroyed `hipGraph_t`.
pub(crate) unsafe fn count_memory_nodes(graph: cubecl_hip_sys::hipGraph_t) -> usize {
    let mut num_nodes: usize = 0;
    // SAFETY: `graph` is a valid `hipGraph_t` per this function's contract. A null node array
    // asks the driver for the node count only, written to `num_nodes`.
    let counted =
        unsafe { cubecl_hip_sys::hipGraphGetNodes(graph, std::ptr::null_mut(), &mut num_nodes) };
    if counted != cubecl_hip_sys::HIP_SUCCESS {
        log::warn!(
            "hipGraphGetNodes failed with HIP status {counted} while counting the graph's \
             nodes; skipping the memory-node check for this capture"
        );
        return 0;
    }
    let mut nodes: Vec<cubecl_hip_sys::hipGraphNode_t> = vec![std::ptr::null_mut(); num_nodes];
    let mut num_read = num_nodes;
    // SAFETY: `graph` is valid per this function's contract, and `nodes` has room for
    // `num_read` entries — the count the call above reported.
    let read =
        unsafe { cubecl_hip_sys::hipGraphGetNodes(graph, nodes.as_mut_ptr(), &mut num_read) };
    if read != cubecl_hip_sys::HIP_SUCCESS {
        log::warn!(
            "hipGraphGetNodes failed with HIP status {read} while reading the graph's \
             {num_nodes} node(s); skipping the memory-node check for this capture"
        );
        return 0;
    }
    nodes
        .iter()
        .take(num_read)
        .filter(|node| {
            let mut ty: cubecl_hip_sys::hipGraphNodeType =
                cubecl_hip_sys::hipGraphNodeType_hipGraphNodeTypeKernel;
            // SAFETY: `node` is one of the handles the driver just wrote into `nodes`, so it
            // is a valid node of the still-live `graph`.
            let queried = unsafe { cubecl_hip_sys::hipGraphNodeGetType(**node, &mut ty) };
            if queried != cubecl_hip_sys::HIP_SUCCESS {
                log::warn!(
                    "hipGraphNodeGetType failed with HIP status {queried}; treating the node \
                     as not a memory node"
                );
                return false;
            }
            matches!(
                ty,
                cubecl_hip_sys::hipGraphNodeType_hipGraphNodeTypeMemAlloc
                    | cubecl_hip_sys::hipGraphNodeType_hipGraphNodeTypeMemFree
            )
        })
        .count()
}

/// Close the capture recording on `sys` and instantiate what it recorded into
/// an executable.
///
/// `doomed` is a recording already known not to seal — a launch inside the
/// window read a buffer carrying a failure, so an operation is missing. The
/// driver capture is closed either way: a stream left in capture mode records
/// every launch that follows it.
///
/// A capture that recorded a memory node is rejected here too. The graph
/// allocates on launch and never frees, so the driver refuses every relaunch
/// while the first quietly succeeds — failing at capture time is what keeps
/// the diagnostic pointing at the cause. What to do about it is the caller's
/// call.
///
/// The intermediate graph is freed on every path, leaving only the executable
/// the caller now owns.
///
/// # Safety
///
/// `sys` must be a valid HIP stream with a capture begun on it.
pub(crate) unsafe fn seal_capture(
    sys: cubecl_hip_sys::hipStream_t,
    doomed: Option<ServerError>,
) -> Result<cubecl_hip_sys::hipGraphExec_t, ServerError> {
    unsafe {
        let mut graph: cubecl_hip_sys::hipGraph_t = std::ptr::null_mut();
        hip_check(
            "hipStreamEndCapture",
            cubecl_hip_sys::hipStreamEndCapture(sys, &mut graph),
        )?;
        if let Some(doomed) = doomed {
            cubecl_hip_sys::hipGraphDestroy(graph);
            return Err(doomed);
        }
        let alloc_nodes = count_memory_nodes(graph);
        if alloc_nodes > 0 {
            cubecl_hip_sys::hipGraphDestroy(graph);
            return Err(ServerError::graph_state(format!(
                "capture recorded {alloc_nodes} memory node(s): an allocation inside the capture \
                 window makes the graph un-relaunchable, so the capture is rejected (the \
                 persistent pool should have served this allocation)"
            )));
        }
        let mut exec: cubecl_hip_sys::hipGraphExec_t = std::ptr::null_mut();
        let instantiated = hip_check(
            "hipGraphInstantiate",
            cubecl_hip_sys::hipGraphInstantiate(
                &mut exec,
                graph,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            ),
        );
        cubecl_hip_sys::hipGraphDestroy(graph);
        instantiated.map(|_| exec)
    }
}
