//! CUDA's half of graph capture: opening the recording, closing it into an
//! executable, staging that executable, and replaying it.
//!
//! Everything around those four — arming the pools, draining the drop queue,
//! pinning what the window touched, claiming what a refused recording would
//! have written — is the shared
//! [`Window`](cubecl_runtime::command::Window)'s.

use crate::compute::driver::Cuda;
use crate::compute::stream::Stream;
use cubecl_environment::backtrace::BackTrace;
use cubecl_runtime::command::GraphDriver;
use cubecl_runtime::server::ServerError;
use cudarc::driver::sys::{CUgraph, CUgraphExec, CUgraphNode, CUresult, CUstream};

/// An instantiated CUDA executable graph, destroyed on drop.
pub struct Executable(CUgraphExec);

impl Drop for Executable {
    fn drop(&mut self) {
        // SAFETY: produced by `cuGraphInstantiateWithFlags` and destroyed
        // exactly once here; the handle is never copied out.
        unsafe {
            cudarc::driver::sys::cuGraphExecDestroy(self.0);
        }
    }
}

impl GraphDriver for Cuda {
    type Executable = Executable;

    fn begin(stream: &mut Stream) -> Result<(), ServerError> {
        // SAFETY: `stream.sys` is a valid CUDA stream; global capture mode
        // records every launch issued on it until `cuStreamEndCapture`.
        let status = unsafe {
            cudarc::driver::sys::cuStreamBeginCapture_v2(
                stream.sys,
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
            )
        };
        checked("cuStreamBeginCapture", status)
    }

    fn instantiate(
        stream: &mut Stream,
        doomed: Option<ServerError>,
    ) -> Result<Executable, ServerError> {
        // SAFETY: a capture was begun on this stream — the shared window's
        // `capturing.end` rejects the call otherwise.
        unsafe { instantiate_recording(stream.sys, doomed) }
    }

    fn upload(exec: &Executable, stream: &mut Stream) {
        // SAFETY: `exec` was instantiated above and `sys` is this stream; the
        // upload is enqueued stream-ordered.
        let uploaded = unsafe { cudarc::driver::sys::cuGraphUpload(exec.0, stream.sys) };
        if let Err(err) = checked("cuGraphUpload", uploaded) {
            log::warn!(
                "Pre-uploading the captured graph failed; \
                 the first replay will upload on demand: {err}"
            );
        }
    }

    fn replay(exec: &Executable, stream: &mut Stream) -> Result<(), ServerError> {
        // SAFETY: `exec` is a valid instantiated graph; launching it on the
        // stream re-runs the recorded sequence.
        let status = unsafe { cudarc::driver::sys::cuGraphLaunch(exec.0, stream.sys) };
        checked("cuGraphLaunch", status)
    }
}

/// Close the capture recording on `sys` and instantiate it into an executable.
///
/// `doomed` is a recording already known not to instantiate — a launch inside
/// the window read a buffer carrying a failure, so an operation is missing.
/// The driver capture is closed either way: a stream left in capture mode
/// records every launch that follows it.
///
/// A capture that recorded a memory node is rejected here too. The graph
/// allocates on launch and never frees, so CUDA refuses every relaunch with
/// `CUDA_ERROR_INVALID_VALUE` while the first quietly succeeds — failing at
/// capture time is what keeps the diagnostic pointing at the cause.
///
/// The intermediate graph is freed on every path, leaving only the executable
/// the caller now owns.
///
/// # Safety
///
/// `sys` must be a valid CUDA stream with a capture begun on it.
unsafe fn instantiate_recording(
    sys: CUstream,
    doomed: Option<ServerError>,
) -> Result<Executable, ServerError> {
    unsafe {
        let mut graph: CUgraph = std::ptr::null_mut();
        checked(
            "cuStreamEndCapture",
            cudarc::driver::sys::cuStreamEndCapture(sys, &mut graph),
        )?;
        if let Some(doomed) = doomed {
            cudarc::driver::sys::cuGraphDestroy(graph);
            return Err(doomed);
        }
        let alloc_nodes = count_memory_nodes(graph);
        if alloc_nodes > 0 {
            cudarc::driver::sys::cuGraphDestroy(graph);
            return Err(ServerError::graph_state(format!(
                "capture recorded {alloc_nodes} memory node(s): an allocation inside the capture \
                 window makes the graph un-relaunchable, so the capture is rejected (the \
                 persistent pool should have served this allocation)"
            )));
        }
        let mut exec: CUgraphExec = std::ptr::null_mut();
        let instantiated = checked(
            "cuGraphInstantiateWithFlags",
            cudarc::driver::sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0),
        );
        cudarc::driver::sys::cuGraphDestroy(graph);
        instantiated.map(|_| Executable(exec))
    }
}

/// Count the memory-allocation/free nodes recorded in `graph`.
///
/// A captured graph is only replayable if it owns no memory nodes: CUDA
/// refuses to relaunch a graph whose allocation nodes have not been freed.
/// Every allocation the capture window needs must therefore be served by the
/// already-warmed persistent pool — the window growing the pool is precisely
/// the condition this detects.
///
/// # Safety
///
/// `graph` must be a valid, not-yet-destroyed `CUgraph`.
unsafe fn count_memory_nodes(graph: CUgraph) -> usize {
    let mut num_nodes: usize = 0;
    // SAFETY: `graph` is a valid `CUgraph` per this function's contract. A null
    // node array asks the driver for the node count only, written to
    // `num_nodes`.
    let counted = unsafe {
        cudarc::driver::sys::cuGraphGetNodes(graph, std::ptr::null_mut(), &mut num_nodes)
    };
    if let Err(err) = checked("cuGraphGetNodes", counted) {
        log::warn!("{err} while counting the graph's nodes; skipping the memory-node check");
        return 0;
    }
    let mut nodes: Vec<CUgraphNode> = vec![std::ptr::null_mut(); num_nodes];
    let mut num_read = num_nodes;
    // SAFETY: `graph` is valid per this function's contract, and `nodes` has
    // room for `num_read` entries — the count the call above reported.
    let read =
        unsafe { cudarc::driver::sys::cuGraphGetNodes(graph, nodes.as_mut_ptr(), &mut num_read) };
    if let Err(err) = checked("cuGraphGetNodes", read) {
        log::warn!(
            "{err} while reading the graph's {num_nodes} node(s); \
             skipping the memory-node check"
        );
        return 0;
    }
    nodes
        .iter()
        .take(num_read)
        .filter(|node| {
            let mut ty = cudarc::driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL;
            // SAFETY: `node` is one of the handles the driver just wrote into
            // `nodes`, so it is a valid node of the still-live `graph`.
            let queried = unsafe { cudarc::driver::sys::cuGraphNodeGetType(**node, &mut ty) };
            if let Err(err) = checked("cuGraphNodeGetType", queried) {
                log::warn!("{err}; treating the node as not a memory node");
                return false;
            }
            matches!(
                ty,
                cudarc::driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_ALLOC
                    | cudarc::driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_FREE
            )
        })
        .count()
}

/// `Ok` when the CUDA driver says the call to `op` succeeded.
///
/// CUDA reports through its own `CUresult` rather than the plain integer the
/// shared [`checked`](cubecl_runtime::driver::checked) takes, and its error
/// values carry their own text, so this is the one place the two diverge.
///
/// # Errors
///
/// [`ServerError::Generic`] naming the entry point and what the driver said.
fn checked(op: &str, status: CUresult) -> Result<(), ServerError> {
    status.result().map_err(|err| ServerError::Generic {
        reason: format!("{op} failed: {err}"),
        backtrace: BackTrace::capture(),
    })
}
