use super::storage::gpu::{GpuResource, GpuStorage};
use crate::{
    compute::{command::Command, context::HipContext, fence::Fence, stream::HipStreamBackend},
    runtime::HipCompiler,
};
use cubecl_common::{bytes::Bytes, profile::ProfileDuration};
use cubecl_core::{
    MemoryConfiguration,
    ir::MemoryDeviceProperties,
    prelude::*,
    server::{
        BufferBinding, CopyDescriptor, Handle, KernelArguments, KernelResource, ProfileError,
        ProfilingToken, ServerCommunication, ServerError, ServerUtilities, StreamErrorMode,
    },
};
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::future;
use cubecl_environment::future::DynFut;
use cubecl_environment::stream::StreamId;
use cubecl_runtime::{
    allocator::PitchedMemoryLayoutPolicy,
    compiler::CubeTask,
    config::{CubeClRuntimeConfig, RuntimeConfig},
    dry_run::LaunchMode,
    id::GraphId,
    logging::ServerLogger,
    memory_management::{
        InstallMemoryPoolsError, ManagedMemoryHandle, ManagedMemoryId, MemoryAllocationMode,
        MemoryReport, MemoryUsage,
    },
    server::ComputeServer,
    storage::{ComputeStorage, ManagedResource},
    stream::MultiStream,
};
use std::collections::HashMap;

use crate::compute::graph::HipGraph;
use std::sync::Arc;

/// Turn a HIP status into a [`ServerError`], naming the failed operation.
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

/// Build — or reuse from the cache — the device buffer holding a launch's info words.
///
/// Reuse a cached info buffer when a launch has already staged these exact info words.
/// The info is read-only metadata (no tensor pointers), so sharing it across launches —
/// even of different kernels — is sound, and it means a stable-shape decode allocates
/// and copies no info inside a capture window (all launches hit warm buffers).
///
/// The cache's policy makes every decision (see
/// [`MetadataInfoCache`](cubecl_runtime::metadata_cache::MetadataInfoCache)), and the
/// capture lifecycle drives its mode so that during capture every buffer is cached and
/// none is evicted. We ask the policy first and only touch the cache when it says to —
/// otherwise we just build the buffer, never keeping a key we wouldn't use. `words` is
/// taken by value so a miss hands it to the cache as the key without cloning. The
/// buffer's bytes always equal the key bytes, so a hit is byte-identical to what the miss
/// path would have built.
fn info_buffer(command: &mut Command<'_>, words: Vec<u64>) -> Result<Handle, ServerError> {
    let size = core::mem::size_of_val(words.as_slice());
    let cache_mode = command.streams.current().capturing.cache_mode();
    command.streams.current().info_cache.mode(cache_mode);

    if !command.streams.current().info_cache.should_cache(size) {
        return Ok(command.create_with_data(bytemuck::cast_slice(&words))?);
    }
    // Look up by the borrowed words — a hit clones nothing. On a miss we build the buffer
    // and move the words into the cache as the key.
    if let Some(handle) = command.streams.current().info_cache.get(&words) {
        return Ok(handle);
    }
    let handle = command.create_with_data(bytemuck::cast_slice(&words))?;
    command
        .streams
        .current()
        .info_cache
        .insert(words, handle.clone());
    Ok(handle)
}

#[derive(Debug)]
pub struct HipServer {
    ctx: HipContext,
    streams: MultiStream<HipStreamBackend>,
    utilities: Arc<ServerUtilities<Self>>,
    /// Captured graphs owned by this server, keyed by the [`GraphId`] handed to
    /// the client. `end_capture` inserts, `replay` looks up, `graph_destroy`
    /// removes (dropping the [`HipGraph`] destroys its executable and unpins the
    /// buffers it retained). Referencing graphs by id keeps the raw
    /// `hipGraphExec_t` inside the server, never boxed across the actor boundary.
    graphs: HashMap<GraphId, HipGraph>,
    /// The memory the launch in flight was given, held across the call that
    /// consumes its arguments so a failure can still name what it left
    /// unwritten. Reused, so a launch allocates nothing for it.
    unwritten: Vec<ManagedMemoryId>,
}

// SAFETY: `HipServer` is only accessed from one thread at a time via the `DeviceHandle`
// (which serializes access through either a mutex or a dedicated runner thread depending
// on the selected channel feature). The HIP context and streams it manages are never
// shared across threads without synchronization.
unsafe impl Send for HipServer {}

impl ComputeServer for HipServer {
    type Kernel = Box<dyn CubeTask<HipCompiler>>;
    type Storage = GpuStorage;
    type MemoryLayoutPolicy = PitchedMemoryLayoutPolicy;
    type Info = ();

    fn logger(&self) -> Arc<ServerLogger> {
        self.streams.logger.clone()
    }

    fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }

    fn staging(&mut self, sizes: &[usize], stream_id: StreamId) -> Result<Vec<Bytes>, ServerError> {
        let mut command = self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        )?;

        Ok(sizes
            .iter()
            .map(|size| command.reserve_cpu(*size, true, None))
            .collect())
    }

    fn initialize_memory(&mut self, memory: ManagedMemoryHandle, size: u64, stream_id: StreamId) {
        let mut command = match self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        ) {
            Ok(val) => val,
            Err(err) => unreachable!("{err}"),
        };

        let reserved = command
            .reserve(size)
            .unwrap_or_else(|err| panic!("failed to reserve {size} bytes of device memory: {err}"));
        command.bind(reserved, memory);
    }

    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        // Buffers another stream wrote are only as good as the work that wrote
        // them; see `StreamPool::ensure_written`.
        if let Err(err) = self
            .streams
            .ensure_written(stream_id, descriptors.iter().map(|d| &d.handle))
        {
            return Box::pin(async move { Err(err) });
        }

        match self.command(
            stream_id,
            descriptors.iter().map(|d| &d.handle),
            StreamErrorMode {
                ignore: false,
                flush: true,
            },
        ) {
            Ok(mut command) => Box::pin(command.read_async(descriptors)),
            Err(err) => Box::pin(async move { Err(err) }),
        }
    }

    fn write(&mut self, descriptors: Vec<(CopyDescriptor, Bytes)>, stream_id: StreamId) {
        let mut command = match self.command(
            stream_id,
            descriptors.iter().map(|desc| &desc.0.handle),
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        ) {
            Ok(val) => val,
            Err(err) => unreachable!("{err}"),
        };

        for (descriptor, data) in descriptors {
            // The copy leaves the destination as it was on failure, which is
            // what a later read of it has to fail on.
            let unwritten = [descriptor.handle.memory.id()];
            if let Err(err) = command.write_to_gpu(descriptor, data) {
                command.error(err.into(), unwritten);
                return;
            }
        }
    }

    unsafe fn launch(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: KernelArguments,
        stream_id: StreamId,
        launch_mode: LaunchMode,
    ) {
        // Named before the launch consumes them: a failure below never reached
        // the device, so every buffer it was given is left as it was, and a read
        // of one has to fail on the error rather than copy it.
        //
        // A dry run stages none. It was never going to write, so a failure in it
        // leaves nothing stale, and naming its buffers would fail unrelated
        // reads of memory the run deliberately left alone.
        self.unwritten.clear();
        if !launch_mode.is_skipped() {
            self.unwritten.extend(bindings.memory_ids_written());
        }

        if let Err(err) = self.launch_checked(kernel, count, bindings, stream_id, launch_mode) {
            let mut stream = match self.streams.resolve(stream_id, [].into_iter(), false) {
                Ok(stream) => stream,
                Err(err) => unreachable!("{err}"),
            };
            stream
                .current()
                .errors
                .push_unwritten(stream_id, err, self.unwritten.iter().copied());
        }
    }

    fn flush(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        let mut command = self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: false,
                flush: true,
            },
        )?;

        let current = command.streams.current();
        current.drop_queue.flush(|| Fence::new(current.sys));
        current.memory_management_gpu.storage().flush();

        Ok(())
    }

    fn graph_prepare(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        let mut command = self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: false,
                flush: true,
            },
        )?;
        let stream = command.streams.current();
        stream.capturing.prepare(stream_id)?;
        // Route every allocation from here until `end_capture` into the
        // persistent pool and snapshot which slices are already in use. Called
        // before the warmup run, so the pool is warm before `begin_capture` —
        // the capture window then reuses those slices with no `hipMalloc`
        // (which would be illegal mid-capture, HIP status 901). `end_capture`
        // pins everything the window added on the graph.
        //
        // Both pools are armed: the GPU pool for tensor and kernel-info buffers,
        // and the pinned CPU pool that stages each kernel's info bytes to the
        // device (a fresh pinned allocation mid-capture would fault the same way).
        stream.memory_management_gpu.capture_begin();
        stream.memory_management_cpu.capture_begin();
        Ok(())
    }

    fn begin_capture(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        let mut command = self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: false,
                flush: true,
            },
        )?;
        let stream = command.streams.current();
        // Rejected before the reclaim below runs: a drop-queue flush issued on
        // a stream that is already recording would abort its live capture.
        stream.capturing.begin()?;
        // Reclaim deferred frees before the capture window opens: warmup's
        // pinned staging buffers (and any other drop-queued slices) sit in the
        // drop queue until flushed, so without this the capture run finds no
        // free staging slice and allocates a fresh one mid-capture — which
        // faults. The queue is a double buffer (a flush only frees the batch
        // from two cycles ago and rotates the current one into `pending`), so
        // flush twice to actually free warmup's just-staged buffers and return
        // them to their pools for the capture run to reuse.
        let sys = stream.sys;
        stream.drop_queue.flush(|| Fence::new(sys));
        stream.drop_queue.flush(|| Fence::new(sys));
        // Warmup is over: release the slices it retained (see `CaptureState::primed`) so the
        // recorded run reuses them instead of allocating. Mandatory rather than an optimization --
        // priming retention is shared runtime behaviour, so leaving it armed here would hold
        // warmup's slices for the whole window and force a mid-capture `hipMalloc`, which
        // invalidates the capture.
        stream.memory_management_gpu.capture_priming_end();
        stream.memory_management_cpu.capture_priming_end();
        // SAFETY: `stream.sys` is a valid HIP stream; global capture mode
        // records every launch issued on it until `hipStreamEndCapture`.
        let status = unsafe {
            cubecl_hip_sys::hipStreamBeginCapture(
                stream.sys,
                cubecl_hip_sys::hipStreamCaptureMode_hipStreamCaptureModeGlobal,
            )
        };
        if let Err(err) = hip_check("hipStreamBeginCapture", status) {
            // The capture never opened: disarm retention, restore the allocation
            // mode, and return to `NoCapture`, so a failed `start_capture`
            // doesn't leave the stream allocating pinned persistent memory
            // forever. The caller can retry the whole
            // `graph_prepare`/`start_capture` sequence.
            stream.memory_management_gpu.capture_end();
            stream.memory_management_cpu.capture_end();
            // Unpin any info-cache entries warmup pinned; the capture is off.
            stream.info_cache.capture_discard();
            stream.capturing.abort();
            return Err(err);
        }
        // Recording now: fenced drop-queue flushes on the execution path are
        // suppressed for the duration of the capture (a host sync would abort
        // it). The deferred staging buffers are reclaimed in `end_capture`.
        Ok(())
    }

    fn end_capture(&mut self, stream_id: StreamId) -> Result<GraphId, ServerError> {
        let id = GraphId::new();
        // Build the graph inside a scope so the `command` borrow of `self` ends
        // before we register the graph in `self.graphs`.
        let hip_graph = {
            // Do NOT flush/surface queued errors here (`ignore: true, flush:
            // false`): this command runs while the stream is still recording, and
            // `flush_errors` would `hipFree` mid-capture — aborting it — and bail
            // via `?` before `hipStreamEndCapture` ever runs, wedging the stream
            // in capture mode forever. Any queued error surfaces on the next
            // normal op once the capture is closed below.
            let mut command = self.command_no_inputs(
                stream_id,
                StreamErrorMode {
                    ignore: true,
                    flush: false,
                },
            )?;
            let stream = command.streams.current();
            // Rejected before `hipStreamEndCapture` runs on a stream that never
            // began a capture. The state leaves capture mode here, so the
            // failure paths below cannot wedge the stream in it — they
            // re-enable the deferred fenced flushes and restore the allocation
            // mode on the way out. A window the caller does not own is closed
            // and torn down all the same, since nobody else is coming back to
            // close it; only its owner gets a graph out of it.
            let outcome = stream.capturing.end(stream_id)?;
            // SAFETY: ends the capture begun on this stream and instantiates the
            // recorded graph into an executable; the intermediate `graph` is freed
            // whether or not instantiation succeeds, leaving only the `exec` the
            // returned handle owns.
            let exec = unsafe {
                let mut graph: cubecl_hip_sys::hipGraph_t = std::ptr::null_mut();
                hip_check(
                    "hipStreamEndCapture",
                    cubecl_hip_sys::hipStreamEndCapture(stream.sys, &mut graph),
                )
                .and_then(|_| {
                    // A capture that recorded a memory node is unusable: the graph allocates on
                    // launch and never frees, so the driver rejects every relaunch while the
                    // first launch quietly succeeds. Fail here instead, so `stop_capture`
                    // surfaces it at capture time — where the diagnostic still points at the
                    // cause — rather than handing back a graph that dies on its second replay.
                    // What to do about it is the caller's call.
                    let alloc_nodes = count_memory_nodes(graph);
                    if alloc_nodes > 0 {
                        cubecl_hip_sys::hipGraphDestroy(graph);
                        return Err(ServerError::graph_state(format!(
                            "capture recorded {alloc_nodes} memory node(s): an allocation inside \
                             the capture window makes the graph un-relaunchable, so the capture \
                             is rejected (the persistent pool should have served this allocation)"
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
                })
            };
            // Pin every buffer the graph touched so the pool never reuses that
            // memory for the graph's lifetime — both the GPU slices and the pinned
            // staging slices the recorded info copies still read from on replay.
            // On failure the handles drop below with `retained`, unpinning them.
            let mut retained = stream.memory_management_gpu.capture_end();
            retained.extend(stream.memory_management_cpu.capture_end());
            // Reclaim the buffers dropped during the capture window, whose fenced
            // flushes were deferred while `capturing` was set. Flush twice: the
            // queue is a double buffer, one flush only rotates the current batch.
            let sys = stream.sys;
            stream.drop_queue.flush(|| Fence::new(sys));
            stream.drop_queue.flush(|| Fence::new(sys));
            // The memory the recorded launches were given. A graph that seals
            // answers for it on a failed replay; one that does not is answered
            // for below, since those launches never ran and now never will.
            let unwritten = stream.capturing.take_recorded();
            // An abandoned window has no graph to hand back: destroy whatever
            // was instantiated and report instead, taking the owner's queued
            // errors along so nothing is left waiting on a logical stream that
            // may never flush again.
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
                    Err(outcome.abandoned_error(stream_id, stream.errors.take(outcome.owner())))
                }
            };
            match exec {
                Ok(exec) => {
                    // Seal the info-cache entries this capture pinned under the
                    // graph's id, so `graph_destroy` can release them later.
                    stream.info_cache.capture_commit(id);
                    // Pre-stage the executable so the first replay doesn't pay
                    // the upload cost. Non-fatal: `hipGraphLaunch` uploads on
                    // demand if this fails. The upload is no guard against
                    // memory nodes, which is why those are rejected above; by
                    // this point the graph is known to have none.
                    // SAFETY: `exec` was instantiated above and `sys` is this
                    // stream; the upload is enqueued stream-ordered.
                    let uploaded = unsafe { cubecl_hip_sys::hipGraphUpload(exec, sys) };
                    if let Err(err) = hip_check("hipGraphUpload", uploaded) {
                        log::warn!(
                            "Pre-uploading the captured graph failed; \
                             the first replay will upload on demand: {err}"
                        );
                    }
                    HipGraph {
                        exec,
                        _retained: retained,
                        unwritten,
                    }
                }
                Err(err) => {
                    // Instantiation failed: unpin the entries this capture pinned
                    // (they stay as ordinary cached values) and drop `retained`.
                    stream.info_cache.capture_discard();
                    // No graph is handed back, so the recorded launches never
                    // run: every buffer they were given is left as it was. The
                    // caller gets the error below; this copy is what makes a
                    // read of one of those buffers fail on some other stream,
                    // which heard nothing.
                    stream.errors.push_returned(err.clone(), unwritten);
                    return Err(err);
                }
            }
        };
        self.graphs.insert(id, hip_graph);
        Ok(id)
    }

    fn replay(&mut self, graph: GraphId, stream_id: StreamId) {
        // Fire-and-forget like `launch`: enqueue the graph dispatch and, on
        // failure, push the error onto the stream's queue so it surfaces on the
        // next flush/sync rather than blocking the caller here.
        if let Err(err) = self.replay_checked(graph, stream_id) {
            // None of the recorded launches ran, so every buffer the graph was
            // captured against is left as it was. An unknown graph names none:
            // the record of which buffers went with it is gone too.
            let unwritten = self
                .graphs
                .get(&graph)
                .map(|hip| hip.unwritten.clone())
                .unwrap_or_default();
            let mut stream = match self.streams.resolve(stream_id, [].into_iter(), false) {
                Ok(stream) => stream,
                Err(err) => unreachable!("{err}"),
            };
            stream
                .current()
                .errors
                .push_unwritten(stream_id, err, unwritten);
        }
    }

    fn graph_destroy(&mut self, graph: GraphId, stream_id: StreamId) {
        // Destroy only after in-flight replays finish: `replay` returns at
        // enqueue time, so a replay may still be running against this executable.
        // No-op for an unknown id (e.g. a double release).
        if !self.graphs.contains_key(&graph) {
            return;
        }
        // Wait for in-flight replays before dropping the executable. A failed
        // sync means the stream already faulted — so no replay is still running
        // against this graph, and destroying is safe — but don't silently
        // swallow the error: surface it on the stream so the next op reports it.
        let synced = cubecl_environment::future::block_on(self.sync(stream_id));
        // `HipGraph::drop` destroys the executable and unpins the buffers it
        // retained.
        self.graphs.remove(&graph);
        if let Ok(mut streams) = self.streams.resolve(stream_id, [].into_iter(), false) {
            let stream = streams.current();
            // Release the info-cache entries this graph pinned; entries no other
            // live graph still pins are dropped, freeing their buffers.
            stream.info_cache.graph_release(graph);
            if let Err(err) = synced {
                stream.errors.push_sync_failure(stream_id, err);
            }
        }
    }

    fn sync(&mut self, stream_id: StreamId) -> DynFut<Result<(), ServerError>> {
        let command = self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: false,
                flush: true,
            },
        );

        match command {
            Ok(mut command) => command.sync(),
            Err(err) => Box::pin(async { Err(err) }),
        }
    }

    fn start_profile(&mut self, stream_id: StreamId) -> Result<ProfilingToken, ServerError> {
        cubecl_environment::future::block_on(self.sync(stream_id))?;
        Ok(self.ctx.timestamps.start())
    }

    fn end_profile(
        &mut self,
        stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        if let Err(err) = cubecl_environment::future::block_on(self.sync(stream_id)) {
            self.ctx
                .timestamps
                .error(ProfileError::Server(Box::new(err)));
        }
        self.ctx.timestamps.stop(token)
    }

    fn get_resource(
        &mut self,
        binding: BufferBinding,
        stream_id: StreamId,
    ) -> Result<ManagedResource<GpuResource>, ServerError> {
        let mut command = self.command(
            stream_id,
            [&binding].into_iter(),
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        )?;
        let memory = binding.memory.clone();
        let resource = command.resource(binding)?;

        Ok(ManagedResource::new(memory, resource))
    }

    fn memory_usage(&mut self, stream_id: StreamId) -> Result<MemoryUsage, ServerError> {
        let mut command = self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: false,
                flush: false,
            },
        )?;
        Ok(command.memory_usage())
    }

    fn memory_report(&mut self, stream_id: StreamId) -> Result<MemoryReport, ServerError> {
        let mut command = self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: false,
                flush: false,
            },
        )?;
        Ok(command.memory_report())
    }

    fn stream_ids(&self) -> Vec<StreamId> {
        self.streams.stream_ids().collect()
    }

    fn memory_cleanup(&mut self, stream_id: StreamId) {
        let mut command = match self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        ) {
            Ok(val) => val,
            // Server is in error.
            Err(_) => return,
        };
        command.memory_cleanup()
    }

    fn allocation_mode(&mut self, mode: MemoryAllocationMode, stream_id: StreamId) {
        let mut command = match self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        ) {
            Ok(val) => val,
            Err(err) => unreachable!("{err}"),
        };
        command.allocation_mode(mode)
    }

    fn install_memory_pools(
        &mut self,
        config: MemoryConfiguration,
        stream_id: StreamId,
    ) -> Result<(), InstallMemoryPoolsError> {
        // Streams created from now on build their GPU pools with the new
        // layout; memory is per stream, so already-created streams keep theirs.
        self.streams.backend_mut().set_gpu_pools(config.clone());
        let (_, props) = self.streams.backend_mut().gpu_pools();

        // The calling stream's pools are rebuilt in place, keeping the old
        // layout when something is still live in them.
        let mut command = match self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        ) {
            Ok(val) => val,
            // Server is in error; the failure itself surfaces at the next sync.
            Err(_) => return Err(InstallMemoryPoolsError::StreamUnavailable),
        };
        command.install_memory_pools(config, &props)
    }
}

impl ServerCommunication for HipServer {
    const SERVER_COMM_ENABLED: bool = false;
}

impl HipServer {
    /// Create a new hip server.
    pub(crate) fn new(
        ctx: HipContext,
        mem_props: MemoryDeviceProperties,
        mem_config: MemoryConfiguration,
        mem_alignment: usize,
        is_integrated: bool,
        utilities: ServerUtilities<Self>,
    ) -> Self {
        let config = CubeClRuntimeConfig::get();
        let max_streams = config.streaming.max_streams;

        Self {
            ctx,
            streams: MultiStream::new(
                utilities.logger.clone(),
                HipStreamBackend::new(
                    mem_props,
                    mem_config,
                    mem_alignment,
                    is_integrated,
                    utilities.logger.clone(),
                ),
                max_streams,
            ),
            utilities: Arc::new(utilities),
            graphs: HashMap::new(),
            unwritten: Vec::new(),
        }
    }

    fn command_no_inputs(
        &mut self,
        stream_id: StreamId,
        mode: StreamErrorMode,
    ) -> Result<Command<'_>, ServerError> {
        self.command(stream_id, [].into_iter(), mode)
    }

    fn command<'a>(
        &mut self,
        stream_id: StreamId,
        handles: impl Iterator<Item = &'a BufferBinding>,
        mode: StreamErrorMode,
    ) -> Result<Command<'_>, ServerError> {
        if mode.flush {
            let errors = self.flush_errors(stream_id);

            if !mode.ignore && !errors.is_empty() {
                return Err(ServerError::ServerUnhealthy {
                    errors,
                    backtrace: BackTrace::capture(),
                });
            }
        }
        let streams = self.streams.resolve(stream_id, handles, !mode.ignore)?;

        Ok(Command::new(&mut self.ctx, streams))
    }

    fn flush_errors(&mut self, stream_id: StreamId) -> Vec<ServerError> {
        let mut stream = match self.streams.resolve(stream_id, [].into_iter(), false) {
            Ok(stream) => stream,
            Err(_) => return Vec::new(),
        };
        let errors = stream.current().errors.take(stream_id);

        // It is very important to tag current profiles as being wrong.
        if !errors.is_empty() {
            self.ctx.timestamps.error(ProfileError::Unknown {
                reason: alloc::format!("{errors:?}"),
                backtrace: BackTrace::capture(),
            });
            stream.current().memory_management_gpu.cleanup(false);
        }

        core::mem::drop(stream);
        errors
    }

    fn launch_checked(
        &mut self,
        kernel: Box<dyn CubeTask<HipCompiler>>,
        count: CubeCount,
        bindings: KernelArguments,
        stream_id: StreamId,
        launch_mode: LaunchMode,
    ) -> Result<(), ServerError> {
        let kernel_id = kernel.id();
        let logger = self.streams.logger.clone();
        let mut command = self.command(
            stream_id,
            bindings.buffers(),
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        )?;

        // A skipped launch stops here, after compilation and before anything
        // that touches a buffer: resolving resources, uploading metadata or
        // reading a dynamic cube count would materialize memory a dry run
        // exists to leave unmapped (and the readback would block on garbage
        // values).
        if launch_mode.is_skipped() {
            command.compile_only(&kernel_id, kernel, logger)?;
            return Ok(());
        }

        // A launch being recorded into a graph hands its buffers to the graph:
        // a replay that fails runs none of the recorded launches, so it leaves
        // all of them as they were. A no-op outside a capture window, and never
        // reached by a dry run, which records nothing to answer for.
        let stream = command.streams.current();
        stream.capturing.record(bindings.memory_ids_written());
        // This launch is on its way, so a capture that failed to seal has
        // nothing left to say about the buffers it is about to write.
        stream.errors.written(bindings.memory_ids_written());

        let count = match count {
            CubeCount::Static(x, y, z) => (x, y, z),
            // TODO: HIP doesn't have an exact equivalent of dynamic dispatch. Instead, kernels are free to launch other kernels.
            // One option is to create a dummy kernel with 1 thread that launches the real kernel with the dynamic dispatch settings.
            // For now, just read the dispatch settings from the buffer.
            CubeCount::Dynamic(binding) => {
                let data = future::block_on(command.read_async(vec![CopyDescriptor::new(
                    binding,
                    [3].into(),
                    [1].into(),
                    4,
                )]))
                .unwrap();
                let data = bytemuck::cast_slice(&data[0]);
                assert!(
                    data.len() == 3,
                    "Dynamic cube count should contain 3 values"
                );
                (data[0], data[1], data[2])
            }
        };

        // A dynamic count can resolve to zero, which the driver rejects.
        if count.0 == 0 || count.1 == 0 || count.2 == 0 {
            return Ok(());
        }

        let KernelArguments { resources, info, .. } = bindings;

        let info_handle = info_buffer(&mut command, info.data)?;

        // Resolving is also where a dry run's deferred allocations get their
        // device backing, so this can fail on a device the measured plan does
        // not fit — reported, not panicked.
        let mut resources = resources
            .into_iter()
            .map(|res| match res {
                KernelResource::Buffer(b) => command.resource(b),
                KernelResource::TensorMap(_) => panic!("Can't use tensor maps on HIP"),
            })
            .collect::<Result<Vec<_>, _>>()?;

        resources.push(command.resource(info_handle.binding())?);

        command.kernel(kernel_id, kernel, count, &resources, logger)?;

        Ok(())
    }

    /// Enqueue a graph replay, returning any error to [`replay`](Self::replay)
    /// to push onto the stream's error queue. Mirrors [`launch_checked`]: the
    /// stream's existing errors are ignored (they surface on the next sync) so a
    /// replay just adds its own on failure.
    ///
    /// [`launch_checked`]: Self::launch_checked
    fn replay_checked(&mut self, graph: GraphId, stream_id: StreamId) -> Result<(), ServerError> {
        // Copy the executable pointer out before borrowing a `command` (which
        // borrows `self`); a raw `hipGraphExec_t` is `Copy`.
        let exec =
            self.graphs
                .get(&graph)
                .map(|hip| hip.exec)
                .ok_or_else(|| ServerError::Generic {
                    reason: "replay was given an unknown or already-destroyed graph".into(),
                    backtrace: BackTrace::capture(),
                })?;
        let mut command = self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        )?;
        let stream = command.streams.current();
        // SAFETY: `exec` is a valid instantiated graph; launching it on the
        // stream re-runs the recorded sequence.
        let status = unsafe { cubecl_hip_sys::hipGraphLaunch(exec, stream.sys) };
        hip_check("hipGraphLaunch", status)
    }

    pub(crate) fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }
}
