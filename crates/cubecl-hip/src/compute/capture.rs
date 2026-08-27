//! HIP's half of graph capture: opening the recording, closing it into an
//! executable, staging that executable, and replaying it.
//!
//! Everything around those four — arming the pools, draining the drop queue,
//! pinning what the window touched, claiming what a refused recording would
//! have written — is the shared
//! [`Window`](cubecl_runtime::command::Window)'s.

use crate::compute::driver::Hip;
use crate::compute::stream::Stream;
use cubecl_hip_sys::{hipGraph_t, hipGraphExec_t};
use cubecl_runtime::command::GraphDriver;
use cubecl_runtime::driver::checked;
use cubecl_runtime::server::ServerError;

/// An instantiated HIP executable graph, destroyed on drop.
pub struct Executable(hipGraphExec_t);

impl Drop for Executable {
    fn drop(&mut self) {
        // SAFETY: produced by `hipGraphInstantiate` and destroyed exactly once
        // here; the handle is never copied out.
        unsafe {
            cubecl_hip_sys::hipGraphExecDestroy(self.0);
        }
    }
}

impl GraphDriver for Hip {
    type Executable = Executable;

    fn begin(stream: &mut Stream) -> Result<(), ServerError> {
        // SAFETY: `stream.sys` is a valid HIP stream; global capture mode
        // records every launch issued on it until `hipStreamEndCapture`.
        let status = unsafe {
            cubecl_hip_sys::hipStreamBeginCapture(
                stream.sys,
                cubecl_hip_sys::hipStreamCaptureMode_hipStreamCaptureModeGlobal,
            )
        };
        Ok(checked("hipStreamBeginCapture", status)?)
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
        let uploaded = unsafe { cubecl_hip_sys::hipGraphUpload(exec.0, stream.sys) };
        if let Err(err) = checked("hipGraphUpload", uploaded) {
            log::warn!(
                "Pre-uploading the captured graph failed; \
                 the first replay will upload on demand: {err}"
            );
        }
    }

    fn replay(exec: &Executable, stream: &mut Stream) -> Result<(), ServerError> {
        // SAFETY: `exec` is a valid instantiated graph; launching it on the
        // stream re-runs the recorded sequence.
        let status = unsafe { cubecl_hip_sys::hipGraphLaunch(exec.0, stream.sys) };
        Ok(checked("hipGraphLaunch", status)?)
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
/// allocates on launch and never frees, so the driver refuses every relaunch
/// while the first quietly succeeds — failing at capture time is what keeps
/// the diagnostic pointing at the cause.
///
/// The intermediate graph is freed on every path, leaving only the executable
/// the caller now owns.
///
/// # Safety
///
/// `sys` must be a valid HIP stream with a capture begun on it.
unsafe fn instantiate_recording(
    sys: cubecl_hip_sys::hipStream_t,
    doomed: Option<ServerError>,
) -> Result<Executable, ServerError> {
    unsafe {
        let mut graph: hipGraph_t = std::ptr::null_mut();
        checked(
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
        let mut exec: hipGraphExec_t = std::ptr::null_mut();
        let instantiated = checked(
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
        instantiated.map(|_| Executable(exec)).map_err(Into::into)
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
unsafe fn count_memory_nodes(graph: cubecl_hip_sys::hipGraph_t) -> usize {
    let mut num_nodes: usize = 0;
    // SAFETY: `graph` is a valid `hipGraph_t` per this function's contract. A null node array
    // asks the driver for the node count only, written to `num_nodes`.
    let counted =
        unsafe { cubecl_hip_sys::hipGraphGetNodes(graph, std::ptr::null_mut(), &mut num_nodes) };
    if let Err(err) = checked("hipGraphGetNodes", counted) {
        log::warn!("{err} while counting the graph's nodes; skipping the memory-node check");
        return 0;
    }
    let mut nodes: Vec<cubecl_hip_sys::hipGraphNode_t> = vec![std::ptr::null_mut(); num_nodes];
    let mut num_read = num_nodes;
    // SAFETY: `graph` is valid per this function's contract, and `nodes` has room for
    // `num_read` entries — the count the call above reported.
    let read =
        unsafe { cubecl_hip_sys::hipGraphGetNodes(graph, nodes.as_mut_ptr(), &mut num_read) };
    if let Err(err) = checked("hipGraphGetNodes", read) {
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
            let mut ty: cubecl_hip_sys::hipGraphNodeType =
                cubecl_hip_sys::hipGraphNodeType_hipGraphNodeTypeKernel;
            // SAFETY: `node` is one of the handles the driver just wrote into `nodes`, so it
            // is a valid node of the still-live `graph`.
            let queried = unsafe { cubecl_hip_sys::hipGraphNodeGetType(**node, &mut ty) };
            if let Err(err) = checked("hipGraphNodeGetType", queried) {
                log::warn!("{err}; treating the node as not a memory node");
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
