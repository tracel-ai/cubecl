//! Graph capture: opening a window on a stream, sealing what it recorded into
//! a replayable executable, and the registry of the executables that sealed.
//!
//! The driver's rules are what shapes this. A stream in capture mode records
//! every launch issued on it, so a window that opened has to be closed even
//! when what it recorded is worthless. Nothing may allocate inside the window,
//! so the pools are warmed before it opens and pinned after it closes. And a
//! host sync would abort the recording, so the fenced flushes the execution
//! path would otherwise run are deferred across it.

use crate::compute::fence::Fence;
use crate::compute::stream::Stream;
use cubecl_core::server::{BufferBinding, ServerError};
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::stream::StreamId;
use cubecl_hip_sys::hipGraphExec_t;
use cubecl_runtime::id::GraphId;
use cubecl_runtime::memory_management::ManagedMemoryHandle;
use std::collections::HashMap;

/// An instantiated HIP executable graph (`hipGraphExec_t`), destroyed on drop.
///
/// Owned by [`Captures`] and referenced by [`GraphId`]; the raw handle never
/// leaves the server actor, which serializes access, so it is only ever
/// touched on the one thread allowed to. The client references the graph by id
/// and, on the last drop, asks the actor to release it — the server syncs the
/// stream before [`Captures::destroy`], so the executable is never destroyed
/// while a replay is still running.
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
/// The graphs this device has sealed, keyed by the [`GraphId`] handed to the
/// client.
///
/// Referencing a graph by id is what keeps the raw `hipGraphExec_t` inside the
/// server: nothing across the actor boundary ever holds one.
#[derive(Debug, Default)]
pub struct Captures {
    graphs: HashMap<GraphId, HipGraph>,
}

impl Captures {
    /// Register a freshly sealed graph under the id its capture was given.
    pub fn insert(&mut self, id: GraphId, graph: HipGraph) {
        self.graphs.insert(id, graph);
    }

    /// Whether `id` still names a live graph.
    pub fn contains(&self, id: GraphId) -> bool {
        self.graphs.contains_key(&id)
    }

    /// The buffers `id`'s recorded launches write.
    ///
    /// Empty for an unknown id, which is the honest answer rather than a
    /// missing one: a graph that is gone took the record of which buffers went
    /// with it, and a replay of it writes nothing.
    pub fn written(&self, id: GraphId) -> Vec<BufferBinding> {
        self.graphs
            .get(&id)
            .map(|graph| graph.written.clone())
            .unwrap_or_default()
    }

    /// Enqueue `id`'s recorded sequence on `stream`.
    ///
    /// The stream's existing errors are ignored — they surface on the next
    /// sync — so a replay only ever adds its own.
    ///
    /// # Errors
    ///
    /// [`ServerError::Generic`] when `id` names no live graph, which the
    /// caller hands straight back: nothing was enqueued and nothing is stale.
    pub fn replay(&self, id: GraphId, stream: &mut Stream) -> Result<(), ServerError> {
        let graph = self.graphs.get(&id).ok_or_else(|| ServerError::Generic {
            reason: "replay was given an unknown or already-destroyed graph".into(),
            backtrace: BackTrace::capture(),
        })?;
        // SAFETY: `exec` is a valid instantiated graph; launching it on the
        // stream re-runs the recorded sequence.
        let status = unsafe { cubecl_hip_sys::hipGraphLaunch(graph.exec, stream.sys) };
        hip_check("hipGraphLaunch", status)
    }

    /// Drop the executable `id` names and release what it held: the buffers it
    /// pinned go with it, and the info-cache entries no other live graph still
    /// pins are freed. A no-op for an unknown id, so a double release is one.
    ///
    /// The caller syncs `stream` first — a replay enqueued against this
    /// executable may still be running.
    pub fn destroy(&mut self, id: GraphId, stream: &mut Stream) {
        self.graphs.remove(&id);
        stream.info_cache.graph_release(id);
    }
}

/// A capture window on one stream: arming the pools before it opens, opening
/// it, and sealing what it recorded.
///
/// The three steps are ordered and the stream refuses them out of order (see
/// [`StreamCapture`](cubecl_runtime::stream::StreamCapture)); this type is
/// where each one's driver-side half lives.
pub struct Window<'a> {
    stream: &'a mut Stream,
}

/// What sealing a capture window produced.
pub enum Sealed {
    /// A replayable executable, with everything the window pinned for it.
    Graph(HipGraph),
    /// The window sealed nothing. Its recorded launches never ran and now
    /// never will, so `written` is left exactly as it was and has to carry
    /// `error` for whoever reads one of those buffers next.
    Refused {
        /// Why the recording could not seal.
        error: ServerError,
        /// The memory those launches would have written.
        written: Vec<BufferBinding>,
    },
}

impl<'a> Window<'a> {
    /// The capture window on `stream`.
    pub fn on(stream: &'a mut Stream) -> Self {
        Self { stream }
    }

    /// Arm the pools for a window about to open, before the warmup run.
    ///
    /// Every allocation from here until the seal is routed into the persistent
    /// pool, and which slices are already in use is snapshotted. The pool is
    /// warm by the time the window opens, so the recorded run reuses those
    /// slices with no `hipMalloc` — which mid-capture is illegal, HIP status
    /// 901. Sealing pins everything the window added on the graph.
    ///
    /// Both pools are armed: the GPU pool for tensor and kernel-info buffers,
    /// and the pinned CPU pool that stages each kernel's info bytes to the
    /// device, where a fresh allocation mid-capture faults the same way.
    pub fn prepare(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        self.stream.capturing.prepare(stream_id)?;
        self.stream.memory_management_gpu.capture_begin();
        self.stream.memory_management_cpu.capture_begin();
        Ok(())
    }

    /// Open the window: from here the driver records every launch issued on
    /// this stream until it is sealed.
    ///
    /// # Errors
    ///
    /// The driver's refusal to begin recording, with the stream left as it was
    /// found — retention disarmed, allocation mode restored, capture state back
    /// to none — so the whole prepare-then-open sequence can simply be retried.
    pub fn begin(&mut self) -> Result<(), ServerError> {
        // Rejected before the reclaim below runs: a drop-queue flush issued on
        // a stream that is already recording would abort its live capture.
        self.stream.capturing.begin()?;
        // Reclaim deferred frees before the window opens: warmup's pinned
        // staging buffers (and any other drop-queued slices) sit in the drop
        // queue until flushed, so without this the recorded run finds no free
        // staging slice and allocates a fresh one mid-capture — which faults.
        // The queue is a double buffer (a flush only frees the batch from two
        // cycles ago and rotates the current one into `pending`), so flush
        // twice to actually free warmup's just-staged buffers and return them
        // to their pools for the recorded run to reuse.
        self.flush_drop_queue_twice();
        // Warmup is over: release the slices it retained (see
        // `CaptureState::primed`) so the recorded run reuses them instead of
        // allocating. Mandatory rather than an optimization — priming
        // retention is shared runtime behaviour, so leaving it armed here would
        // hold warmup's slices for the whole window and force a mid-capture
        // `hipMalloc`, which invalidates the capture.
        self.stream.memory_management_gpu.capture_priming_end();
        self.stream.memory_management_cpu.capture_priming_end();
        // SAFETY: `stream.sys` is a valid HIP stream; global capture mode
        // records every launch issued on it until `hipStreamEndCapture`.
        let status = unsafe {
            cubecl_hip_sys::hipStreamBeginCapture(
                self.stream.sys,
                cubecl_hip_sys::hipStreamCaptureMode_hipStreamCaptureModeGlobal,
            )
        };
        if let Err(err) = hip_check("hipStreamBeginCapture", status) {
            self.stream.memory_management_gpu.capture_end();
            self.stream.memory_management_cpu.capture_end();
            self.stream.info_cache.capture_discard();
            self.stream.capturing.abort();
            return Err(err);
        }
        // Recording now: fenced drop-queue flushes on the execution path are
        // suppressed for as long as the window is open, since a host sync
        // would abort it. The deferred buffers are reclaimed by the seal.
        Ok(())
    }

    /// Close the window and turn what it recorded into a graph registered
    /// under `id`.
    ///
    /// The window leaves capture mode first, so none of the paths below can
    /// wedge the stream in it — they re-enable the deferred fenced flushes and
    /// restore the allocation mode on the way out. A window the caller does
    /// not own is closed and torn down all the same, since nobody else is
    /// coming back to close it; only its owner gets a graph out of it.
    ///
    /// # Errors
    ///
    /// Only when there was no window to seal. A window that opened always
    /// closes: whether it produced anything is [`Sealed`]'s answer, not an
    /// error, because the memory its launches would have written is the
    /// caller's to claim.
    pub fn seal(&mut self, stream_id: StreamId, id: GraphId) -> Result<Sealed, ServerError> {
        let outcome = self.stream.capturing.end(stream_id)?;
        // A launch inside the window read a buffer carrying a failure, so the
        // recording is missing an operation and must not seal; the driver
        // capture still has to be closed either way.
        let doomed = self.stream.capturing.take_failure().map(|reason| {
            ServerError::graph_state(format!(
                "capture recorded a launch whose input carried a failure, so the recording \
                 is missing an operation and cannot seal: {reason}"
            ))
        });
        // SAFETY: a capture was begun on this stream — `capturing.end` above
        // rejects the call otherwise.
        let exec = unsafe { seal_capture(self.stream.sys, doomed.clone()) };
        // Pin every buffer the window touched so the pool never reuses that
        // memory for the graph's lifetime — both the GPU slices and the pinned
        // staging slices the recorded info copies still read from on replay. On
        // failure the handles drop with `retained`, unpinning them.
        let mut retained = self.stream.memory_management_gpu.capture_end();
        retained.extend(self.stream.memory_management_cpu.capture_end());
        // Reclaim the buffers dropped while the window was open, whose fenced
        // flushes were deferred for as long as it was.
        self.flush_drop_queue_twice();
        // The memory the recorded launches write. A graph that seals answers
        // for it on a failed replay; one that does not is answered for by the
        // caller, since those launches never ran and now never will.
        let written = self.stream.capturing.take_recorded();
        // An abandoned window has no graph to hand back: destroy whatever was
        // instantiated and report instead, carrying along whatever had already
        // doomed the recording so the caller sees both reasons.
        let exec = match outcome.is_abandoned() {
            false => exec,
            true => {
                if let Ok(exec) = exec {
                    // SAFETY: instantiated just above, destroyed exactly once
                    // here, and never handed out.
                    unsafe {
                        cubecl_hip_sys::hipGraphExecDestroy(exec);
                    }
                }
                Err(outcome.abandoned_error(stream_id, doomed))
            }
        };
        Ok(match exec {
            Ok(exec) => {
                // Seal the info-cache entries this window pinned under the
                // graph's id, so destroying it can release them later.
                self.stream.info_cache.capture_commit(id);
                self.pre_upload(exec);
                Sealed::Graph(HipGraph {
                    exec,
                    _retained: retained,
                    written,
                })
            }
            Err(error) => {
                // Unpin the entries this window pinned — they stay as ordinary
                // cached values — and drop `retained`.
                self.stream.info_cache.capture_discard();
                Sealed::Refused { error, written }
            }
        })
    }

    /// Pre-stage `exec` so the first replay does not pay the upload cost.
    ///
    /// Non-fatal: `hipGraphLaunch` uploads on demand if this fails. The upload
    /// is no guard against memory nodes, which is why those are rejected at
    /// instantiation; by this point the graph is known to have none.
    fn pre_upload(&mut self, exec: hipGraphExec_t) {
        // SAFETY: `exec` was just instantiated and `sys` is this stream; the
        // upload is enqueued stream-ordered.
        let uploaded = unsafe { cubecl_hip_sys::hipGraphUpload(exec, self.stream.sys) };
        if let Err(err) = hip_check("hipGraphUpload", uploaded) {
            log::warn!(
                "Pre-uploading the captured graph failed; \
                 the first replay will upload on demand: {err}"
            );
        }
    }

    /// Drain the drop queue for real. It is a double buffer — one flush frees
    /// the batch from two cycles ago and rotates the current one into pending —
    /// so a single flush would leave the slices just staged still held.
    fn flush_drop_queue_twice(&mut self) {
        let sys = self.stream.sys;
        self.stream.drop_queue.flush(|| Fence::new(sys));
        self.stream.drop_queue.flush(|| Fence::new(sys));
    }
}

fn hip_check(op: &str, status: cubecl_hip_sys::hipError_t) -> Result<(), ServerError> {
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
unsafe fn count_memory_nodes(graph: cubecl_hip_sys::hipGraph_t) -> usize {
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
unsafe fn seal_capture(
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
