use super::storage::gpu::{GpuResource, GpuStorage};
use crate::{
    CudaCompiler,
    compute::{
        command::Command,
        communication::{get_nccl_comm_id, get_nccl_dtype_count, to_nccl_op},
        context::CudaContext,
        graph::CudaGraph,
        stream::{CudaStreamBackend, StreamCaptureState},
        sync::Fence,
    },
};
use cubecl_common::{bytes::Bytes, profile::ProfileDuration};
use cubecl_core::{
    MemoryConfiguration,
    device::DeviceId,
    ir::{ElemType, FloatKind, IntKind, MemoryDeviceProperties, StorageType, UIntKind},
    prelude::*,
    server::{
        Binding, CommunicationId, CopyDescriptor, Handle, KernelArguments, LaunchError,
        ProfileError, ProfilingToken, ReduceOperation, ServerCommunication, ServerError,
        ServerUtilities, StreamErrorMode, TensorMapBinding, TensorMapMeta,
    },
};
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::future::{self, DynFut};
use cubecl_environment::stream::StreamId;
use cubecl_runtime::{
    allocator::PitchedMemoryLayoutPolicy,
    compiler::CubeTask,
    config::{CubeClRuntimeConfig, RuntimeConfig},
    dry_run::LaunchMode,
    id::GraphId,
    logging::ServerLogger,
    memory_management::{ManagedMemoryHandle, MemoryAllocationMode, MemoryUsage},
    server::ComputeServer,
    storage::{ComputeStorage, ManagedResource},
    stream::MultiStream,
};
use cudarc::driver::sys::{
    CUstream_st, CUtensorMapDataType, CUtensorMapFloatOOBfill, CUtensorMapInterleave,
    CUtensorMapL2promotion, CUtensorMapSwizzle, cuTensorMapEncodeIm2col, cuTensorMapEncodeTiled,
};
use std::{
    collections::{HashMap, hash_map::Entry},
    ffi::c_void,
    mem::MaybeUninit,
    sync::Arc,
};

pub(crate) const MB: usize = 1024 * 1024;

/// Turn a CUDA driver status into a [`ServerError`], naming the failed operation.
fn cuda_check(op: &str, status: cudarc::driver::sys::CUresult) -> Result<(), ServerError> {
    status.result().map_err(|err| ServerError::Generic {
        reason: format!("{op} failed: {err}"),
        backtrace: BackTrace::capture(),
    })
}

/// Count the memory-allocation/free nodes recorded in `graph`.
///
/// A captured graph is only replayable if it owns no memory nodes: CUDA refuses to relaunch a
/// graph whose allocation nodes have not been freed. Every allocation the capture window needs
/// must therefore be served by the already-warmed persistent pool — the window growing the pool
/// is precisely the condition this detects.
///
/// # Safety
///
/// `graph` must be a valid, not-yet-destroyed `CUgraph`.
unsafe fn count_memory_nodes(graph: cudarc::driver::sys::CUgraph) -> usize {
    let mut num_nodes: usize = 0;
    // SAFETY: `graph` is a valid `CUgraph` per this function's contract. A null node array
    // asks the driver for the node count only, written to `num_nodes`.
    let counted = unsafe {
        cudarc::driver::sys::cuGraphGetNodes(graph, std::ptr::null_mut(), &mut num_nodes)
    };
    if counted != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        log::warn!(
            "cuGraphGetNodes failed ({counted:?}) while counting the graph's nodes; \
             skipping the memory-node check for this capture"
        );
        return 0;
    }
    let mut nodes: Vec<cudarc::driver::sys::CUgraphNode> = vec![std::ptr::null_mut(); num_nodes];
    let mut num_read = num_nodes;
    // SAFETY: `graph` is valid per this function's contract, and `nodes` has room for
    // `num_read` entries — the count the call above reported.
    let read =
        unsafe { cudarc::driver::sys::cuGraphGetNodes(graph, nodes.as_mut_ptr(), &mut num_read) };
    if read != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        log::warn!(
            "cuGraphGetNodes failed ({read:?}) while reading the graph's {num_nodes} node(s); \
             skipping the memory-node check for this capture"
        );
        return 0;
    }
    nodes
        .iter()
        .take(num_read)
        .filter(|node| {
            let mut ty = cudarc::driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL;
            // SAFETY: `node` is one of the handles the driver just wrote into `nodes`, so it
            // is a valid node of the still-live `graph`.
            let queried = unsafe { cudarc::driver::sys::cuGraphNodeGetType(**node, &mut ty) };
            if queried != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                log::warn!(
                    "cuGraphNodeGetType failed ({queried:?}); treating the node as not a \
                     memory node"
                );
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

/// Build a [`ServerError`] for a graph-capture call issued in the wrong state
/// (e.g. `begin_capture` without `graph_prepare`, or a second overlapping
/// capture on the same stream).
fn graph_state_error(reason: impl Into<String>) -> ServerError {
    ServerError::Generic {
        reason: reason.into(),
        backtrace: BackTrace::capture(),
    }
}

/// Stage `words` into a device buffer, reusing a cached one when a launch has
/// already staged these exact info words. The info is read-only metadata (no
/// tensor pointers), so sharing it across launches — even of different kernels
/// — is sound, and it means a stable-shape decode allocates and copies no info
/// inside a capture window (all launches hit warm buffers, so the captured
/// graph gains no memcpy nodes for them).
///
/// The cache's policy makes every decision (see
/// [`MetadataInfoCache`](cubecl_runtime::metadata_cache::MetadataInfoCache)),
/// and the capture lifecycle drives its mode so that during capture every
/// buffer is cached and none is evicted. We ask the policy first and only touch
/// the cache when it says to — otherwise we just build the buffer, never
/// cloning a key we wouldn't keep. The buffer's bytes always equal the key
/// bytes, so a hit is byte-identical to what the miss path would have built.
fn info_buffer(command: &mut Command<'_>, words: &[u64]) -> Result<Handle, ServerError> {
    let size = core::mem::size_of_val(words);
    let cache_mode = command.streams.current().capturing.cache_mode();
    command.streams.current().info_cache.mode(cache_mode);

    if !command.streams.current().info_cache.should_cache(size) {
        return Ok(command.create_with_data(bytemuck::cast_slice(words))?);
    }
    // Look up by the borrowed words — a hit clones nothing. On a miss we build
    // the buffer and clone the words into the cache as the key.
    if let Some(handle) = command.streams.current().info_cache.get(words) {
        return Ok(handle);
    }
    let handle = command.create_with_data(bytemuck::cast_slice(words))?;
    command
        .streams
        .current()
        .info_cache
        .insert(words.to_vec(), handle.clone());
    Ok(handle)
}

#[derive(Debug)]
pub struct CudaServer {
    ctx: CudaContext,
    device_id: DeviceId,
    streams: MultiStream<CudaStreamBackend>,
    utilities: Arc<ServerUtilities<Self>>,
    comm_stream: *mut CUstream_st,
    communicators: HashMap<CommunicationId, *mut cudarc::nccl::sys::ncclComm>,
    /// Captured graphs owned by this server, keyed by the [`GraphId`] handed to
    /// the client. `end_capture` inserts, `replay` looks up, `graph_destroy`
    /// removes (dropping the [`CudaGraph`] destroys its executable and unpins the
    /// buffers it retained). Referencing graphs by id keeps the raw
    /// `CUgraphExec` inside the server, never boxed across the actor boundary.
    graphs: HashMap<GraphId, CudaGraph>,
}

// SAFETY: `CudaServer` is only accessed from one thread at a time via the `DeviceHandle`,
// which serializes all server access. The CUDA context, streams, and NCCL communicators
// it manages are never shared across threads without synchronization.
unsafe impl Send for CudaServer {}

impl ComputeServer for CudaServer {
    type Kernel = Box<dyn CubeTask<CudaCompiler>>;
    type Storage = GpuStorage;
    type MemoryLayoutPolicy = PitchedMemoryLayoutPolicy;
    type Info = ();

    fn logger(&self) -> Arc<ServerLogger> {
        self.streams.logger.clone()
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

    fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }

    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>> {
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

        let reserved = command.reserve(size).unwrap();
        command.bind(reserved, memory);
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
            if let Err(err) = command.write_to_gpu(descriptor, data) {
                command.error(err.into());
                return;
            }
        }
    }

    unsafe fn launch(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: KernelArguments,
        mode: ExecutionMode,
        stream_id: StreamId,
        launch_mode: LaunchMode,
    ) {
        if let Err(err) = self.launch_checked(kernel, count, bindings, mode, stream_id, launch_mode)
        {
            let mut stream = match self.streams.resolve(stream_id, [].into_iter(), false) {
                Ok(stream) => stream,
                Err(err) => unreachable!("{err}"),
            };
            stream.current().errors.push(err);
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
        // A capture must be prepared exactly once before it starts; reject a
        // second prepare or a prepare over a live capture so two captures can
        // never overlap on one stream.
        match stream.capturing {
            StreamCaptureState::NoCapture => {}
            StreamCaptureState::Prepare => {
                return Err(graph_state_error(
                    "graph_prepare: a graph capture is already prepared on this stream",
                ));
            }
            StreamCaptureState::Capture => {
                return Err(graph_state_error(
                    "graph_prepare: a graph capture is already recording on this stream",
                ));
            }
        }
        // Route every allocation from here until `end_capture` into the
        // persistent pool and snapshot which slices are already in use. Called
        // before the warmup run, so the pool is warm before `begin_capture` —
        // the capture window then reuses those slices with no `cuMemAlloc`
        // (which would be illegal mid-capture,
        // `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`). `end_capture` pins
        // everything the window added on the graph.
        //
        // Both pools are armed: the GPU pool for tensor and kernel-info buffers,
        // and the pinned CPU pool that stages each kernel's info bytes to the
        // device (a fresh pinned allocation mid-capture would fault the same way).
        stream.memory_management_gpu.capture_begin();
        stream.memory_management_cpu.capture_begin();
        stream.capturing = StreamCaptureState::Prepare;
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
        // A capture must be armed by `graph_prepare` first: the persistent pool
        // it primes (warmed by the run between prepare and here) is what lets
        // the window reuse slices with no illegal mid-capture `cuMemAlloc`.
        // Reject an unprepared start, and reject a second start over a live
        // capture, so captures never overlap on one stream.
        match stream.capturing {
            StreamCaptureState::Prepare => {}
            StreamCaptureState::NoCapture => {
                return Err(graph_state_error(
                    "begin_capture: call graph_prepare before starting a capture",
                ));
            }
            StreamCaptureState::Capture => {
                return Err(graph_state_error(
                    "begin_capture: a graph capture is already recording on this stream",
                ));
            }
        }
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
        // Warmup is over: release the slices it retained (see `CaptureState::primed`) so they are
        // free for the recorded run to reuse. The pool now holds warmup's full distinct working
        // set rather than its transient peak, so the window has nothing left to allocate — and an
        // allocation inside the window would record a memory node, which makes the graph
        // un-relaunchable. Must happen here, after warmup and before the window opens.
        stream.memory_management_gpu.capture_priming_end();
        stream.memory_management_cpu.capture_priming_end();
        // SAFETY: `stream.sys` is a valid CUDA stream; global capture mode
        // records every launch issued on it until `cuStreamEndCapture`.
        let status = unsafe {
            cudarc::driver::sys::cuStreamBeginCapture_v2(
                stream.sys,
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
            )
        };
        if let Err(err) = cuda_check("cuStreamBeginCapture", status) {
            // The capture never opened: disarm retention, restore the allocation
            // mode, and return to `NoCapture`, so a failed `start_capture`
            // doesn't leave the stream allocating pinned persistent memory
            // forever. The caller can retry the whole
            // `graph_prepare`/`start_capture` sequence.
            stream.memory_management_gpu.capture_end();
            stream.memory_management_cpu.capture_end();
            // Unpin any info-cache entries warmup pinned; the capture is off.
            stream.info_cache.capture_discard();
            stream.capturing = StreamCaptureState::NoCapture;
            return Err(err);
        }
        // Recording now; suppress fenced drop-queue flushes on the execution
        // path for the duration of the capture (a host sync would abort it).
        // The deferred staging buffers are reclaimed in `end_capture`.
        stream.capturing = StreamCaptureState::Capture;
        Ok(())
    }

    fn end_capture(&mut self, stream_id: StreamId) -> Result<GraphId, ServerError> {
        let id = GraphId::new();
        // Build the graph inside a scope so the `command` borrow of `self` ends
        // before we register the graph in `self.graphs`.
        let cuda_graph = {
            // Do NOT flush/surface queued errors here (`ignore: true, flush:
            // false`): this command runs while the stream is still recording, and
            // `flush_errors` would free memory mid-capture — aborting it — and
            // bail via `?` before `cuStreamEndCapture` ever runs, wedging the
            // stream in capture mode forever. Any queued error surfaces on the
            // next normal op once the capture is closed below.
            let mut command = self.command_no_inputs(
                stream_id,
                StreamErrorMode {
                    ignore: true,
                    flush: false,
                },
            )?;
            let stream = command.streams.current();
            // Only a recording stream can be ended; reject a stray `end_capture`
            // (nothing prepared/started, or the capture already ended) instead of
            // calling `cuStreamEndCapture` on a stream that never began one.
            if !stream.capturing.is_recording() {
                return Err(graph_state_error(
                    "end_capture: no graph capture is recording on this stream",
                ));
            }
            // SAFETY: ends the capture begun on this stream and instantiates the
            // recorded graph into an executable; the intermediate `graph` is freed
            // whether or not instantiation succeeds, leaving only the `exec` the
            // returned handle owns.
            let exec = unsafe {
                let mut graph: cudarc::driver::sys::CUgraph = std::ptr::null_mut();
                cuda_check(
                    "cuStreamEndCapture",
                    cudarc::driver::sys::cuStreamEndCapture(stream.sys, &mut graph),
                )
                .and_then(|_| {
                    // A capture that recorded a memory node is unusable: the graph allocates on
                    // launch and never frees, so the driver rejects every relaunch with
                    // `CUDA_ERROR_INVALID_VALUE` while the first launch quietly succeeds. Fail
                    // here instead, so `stop_capture` surfaces it at capture time — where the
                    // diagnostic still points at the cause — rather than handing back a graph
                    // that dies on its second replay. What to do about it is the caller's call.
                    let alloc_nodes = count_memory_nodes(graph);
                    if alloc_nodes > 0 {
                        cudarc::driver::sys::cuGraphDestroy(graph);
                        return Err(graph_state_error(format!(
                            "capture recorded {alloc_nodes} memory node(s): an allocation inside \
                             the capture window makes the graph un-relaunchable, so the capture \
                             is rejected (the persistent pool should have served this allocation)"
                        )));
                    }
                    let mut exec: cudarc::driver::sys::CUgraphExec = std::ptr::null_mut();
                    let instantiated = cuda_check(
                        "cuGraphInstantiateWithFlags",
                        cudarc::driver::sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0),
                    );
                    cudarc::driver::sys::cuGraphDestroy(graph);
                    instantiated.map(|_| exec)
                })
            };
            // The capture is over even if it failed to instantiate: re-enable the
            // deferred fenced flushes and restore the allocation mode, so an error
            // here doesn't leave the stream stuck in capture/persistent state.
            stream.capturing = StreamCaptureState::NoCapture;
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
            match exec {
                Ok(exec) => {
                    // Seal the info-cache entries this capture pinned under the
                    // graph's id, so `graph_destroy` can release them later.
                    stream.info_cache.capture_commit(id);
                    // Pre-stage the executable so the first replay doesn't pay
                    // the upload cost. Non-fatal: `cuGraphLaunch` uploads on
                    // demand if this fails. The upload is no guard against memory
                    // nodes — it returns `CUDA_SUCCESS` even for graphs holding
                    // them, which is why those are rejected above; by this point
                    // the graph is known to have none.
                    // SAFETY: `exec` was instantiated above and `sys` is this
                    // stream; the upload is enqueued stream-ordered.
                    let uploaded = unsafe { cudarc::driver::sys::cuGraphUpload(exec, sys) };
                    if let Err(err) = cuda_check("cuGraphUpload", uploaded) {
                        log::warn!(
                            "Pre-uploading the captured graph failed; \
                             the first replay will upload on demand: {err}"
                        );
                    }
                    CudaGraph {
                        exec,
                        _retained: retained,
                    }
                }
                Err(err) => {
                    // Instantiation failed: unpin the entries this capture pinned
                    // (they stay as ordinary cached values) and drop `retained`.
                    stream.info_cache.capture_discard();
                    return Err(err);
                }
            }
        };
        self.graphs.insert(id, cuda_graph);
        Ok(id)
    }

    fn replay(&mut self, graph: GraphId, stream_id: StreamId) {
        // Fire-and-forget like `launch`: enqueue the graph dispatch and, on
        // failure, push the error onto the stream's queue so it surfaces on the
        // next flush/sync rather than blocking the caller here.
        if let Err(err) = self.replay_checked(graph, stream_id) {
            let mut stream = match self.streams.resolve(stream_id, [].into_iter(), false) {
                Ok(stream) => stream,
                Err(err) => unreachable!("{err}"),
            };
            stream.current().errors.push(err);
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
        // `CudaGraph::drop` destroys the executable and unpins the buffers it
        // retained.
        self.graphs.remove(&graph);
        if let Ok(mut streams) = self.streams.resolve(stream_id, [].into_iter(), false) {
            let stream = streams.current();
            // Release the info-cache entries this graph pinned; entries no other
            // live graph still pins are dropped, freeing their buffers.
            stream.info_cache.graph_release(graph);
            if let Err(err) = synced {
                stream.errors.push(err);
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
        binding: Binding,
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
            Err(err) => unreachable!("{err}"),
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

    fn configure_memory_pools(&mut self, config: MemoryConfiguration, stream_id: StreamId) -> bool {
        // Streams created from now on build their GPU pools with the new
        // layout; memory is per stream, so already-created streams keep theirs.
        self.streams.backend_mut().set_gpu_pools(config.clone());
        let (_, props) = self.streams.backend_mut().gpu_pools();

        // The calling stream's pools are rebuilt in place (kept, with a log,
        // when something is still live in them).
        let mut command = match self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        ) {
            Ok(val) => val,
            // Server is in error.
            Err(_) => return false,
        };
        command.configure_memory_pools(config, &props)
    }
}

impl ServerCommunication for CudaServer {
    const SERVER_COMM_ENABLED: bool = true;

    fn comm_init(&mut self, device_ids: Vec<DeviceId>) -> Result<(), ServerError> {
        let id = CommunicationId::from(device_ids.clone());
        if let Entry::Vacant(e) = self.communicators.entry(id.clone()) {
            let mut comm = MaybeUninit::uninit();
            let mut device_ids = device_ids.clone();
            device_ids.sort();
            let rank = device_ids
                .iter()
                .position(|id| id.index_id == self.device_id.index_id)
                .expect("Device's peer id should be in the list of device ids.");
            let nccl_comm_id = get_nccl_comm_id(device_ids.clone());

            // SAFETY: `comm` is a valid `MaybeUninit`. `nccl_comm_id` is a unique communicator ID
            // shared across all participating ranks. `rank` is this device's position in the
            // group. `comm_init_rank` initializes the communicator, making `assume_init` valid.
            unsafe {
                cudarc::nccl::result::comm_init_rank(
                    comm.as_mut_ptr(),
                    device_ids.len() as i32,
                    nccl_comm_id,
                    rank as i32,
                )
                .map_err(|e| ServerError::Generic {
                    reason: format!("NCCL comm_init_rank failed: {e:?}"),
                    backtrace: BackTrace::capture(),
                })?;
                e.insert(comm.assume_init());
            }

            let mut initialized_comms = self.utilities.initialized_comms.write();
            initialized_comms.insert(id);
        }

        Ok(())
    }

    fn all_reduce(
        &mut self,
        src: Binding,
        dst: Binding,
        dtype: ElemType,
        stream_id: StreamId,
        op: ReduceOperation,
        device_ids: Vec<DeviceId>,
    ) -> Result<(), ServerError> {
        // We create a command on the server to retrieve the correct resource of the source and the destination
        // from the memory pools.
        if src.stream != dst.stream {
            for stream in [src.stream, dst.stream].iter() {
                let mut command = self.command_no_inputs(
                    *stream,
                    StreamErrorMode {
                        ignore: false,
                        flush: false,
                    },
                )?;
                command.error(ServerError::Generic {
                    reason: "Source and destination should be on the same stream.".into(),
                    backtrace: BackTrace::capture(),
                });
            }
        }

        let mut command_src = self.command(
            stream_id,
            [&src, &dst].into_iter(),
            StreamErrorMode {
                ignore: false,
                flush: false,
            },
        )?;
        let resource_src = command_src.resource(src)?;
        let resource_dst = command_src.resource(dst)?;

        let stream = command_src.streams.current().sys;

        // We need to free the command before accessing communicators.
        core::mem::drop(command_src);

        // Wait for data to be ready on compute stream.
        Fence::new(stream).wait_async(self.comm_stream);

        // Get the communicator.
        let comm = self
            .communicators
            .get(&CommunicationId::from(device_ids))
            .expect("Communicator for this ID should be initialized");

        // Perform the `cudarc::nccl::result::all_reduce` operation.
        let (nccl_dtype, count) = get_nccl_dtype_count(dtype, resource_src.size);
        // SAFETY: `resource_src.ptr` and `resource_dst.ptr` are valid device pointers.
        // `comm` is a valid NCCL communicator initialized via `comm_init_rank`.
        // `self.comm_stream` is a valid CUDA stream dedicated to collective operations.

        unsafe {
            cudarc::nccl::result::all_reduce(
                resource_src.ptr as *const _,
                resource_dst.ptr as *mut _,
                count,
                nccl_dtype,
                to_nccl_op(op),
                *comm,
                self.comm_stream as _,
            )
            .map_err(|e| ServerError::Generic {
                reason: format!("NCCL all_reduce failed: {e:?}"),
                backtrace: BackTrace::capture(),
            })?;
        }

        Ok(())
    }

    fn sync_collective(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        let mut command = self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        )?;
        let stream = command.streams.current().sys;

        drop(command);

        Fence::new(self.comm_stream).wait_async(stream);

        Ok(())
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(desc)))]
    fn send(
        &mut self,
        desc: CopyDescriptor,
        dtype: ElemType,
        stream_id: StreamId,
        device_id_dst: DeviceId,
    ) -> Result<(), ServerError> {
        let binding = desc.handle.clone();

        // We create a command on the source server to retrieve the correct resource from the
        // source memory pools. We also make sure the current stream is aligned with the stream of
        // the binding, where the data was first allocated.
        let mut command = self.command(
            stream_id,
            [&desc.handle].into_iter(),
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        )?;
        let resource = command.resource(binding.clone())?;
        let stream = command.streams.current().sys;

        // We need to free the command before creating another one.
        core::mem::drop(command);

        // Wait for data to be ready on compute stream.
        Fence::new(stream).wait_async(self.comm_stream);

        // Get the communicator.
        let mut device_ids = vec![device_id_dst, self.device_id];
        device_ids.sort();
        let comm_id = CommunicationId::from(device_ids.clone());
        let comm = self
            .communicators
            .get(&comm_id)
            .expect("Communicator for this ID should exist");

        let rank_dst = device_ids
            .iter()
            .position(|id| id.index_id != self.device_id.index_id)
            .unwrap() as i32;

        // Perform the `send` operation.
        let (nccl_dtype, count) = get_nccl_dtype_count(dtype, resource.size);
        // SAFETY: `resource.ptr` is a valid device pointer.
        // `comm` is a valid NCCL communicator initialized via `comm_init_rank`.
        // `self.comm_stream` is a valid CUDA stream dedicated to collective operations.
        unsafe {
            cudarc::nccl::result::send(
                resource.ptr as *const _,
                count,
                nccl_dtype,
                rank_dst,
                *comm,
                self.comm_stream as _,
            )
            .map_err(|e| ServerError::Generic {
                reason: format!("NCCL send failed: {e:?}"),
                backtrace: BackTrace::capture(),
            })?;
        }

        Ok(())
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace"))]
    fn recv(
        &mut self,
        handle: Handle,
        dtype: ElemType,
        stream_id: StreamId,
        device_id_src: DeviceId,
    ) -> Result<(), ServerError> {
        // We create a new command on the destination server to reserve the necessary GPU memory.
        let mut command_dst = self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        )?;

        let memory = command_dst.reserve(handle.size()).unwrap();
        command_dst.bind(memory, handle.memory.clone());

        let resource_dst = command_dst.resource(handle.binding())?;

        core::mem::drop(command_dst);

        // Get the communicator.
        let mut device_ids = vec![device_id_src, self.device_id];
        device_ids.sort();
        let comm_id = CommunicationId::from(device_ids.clone());
        let comm = self
            .communicators
            .get(&comm_id)
            .expect("Communicator for this ID should exist");

        let rank_src = device_ids
            .iter()
            .position(|id| id.index_id != self.device_id.index_id)
            .unwrap() as i32;

        // Perform the `recv` operation.
        let (nccl_dtype, count) = get_nccl_dtype_count(dtype, resource_dst.size);
        // SAFETY: `resource.ptr` is a valid device pointer.
        // `comm` is a valid NCCL communicator initialized via `comm_init_rank`.
        // `self.comm_stream` is a valid CUDA stream dedicated to collective operations.
        unsafe {
            cudarc::nccl::result::recv(
                resource_dst.ptr as *mut _,
                count,
                nccl_dtype,
                rank_src,
                *comm,
                self.comm_stream as _,
            )
            .map_err(|e| ServerError::Generic {
                reason: format!("NCCL recv failed: {e:?}"),
                backtrace: BackTrace::capture(),
            })?;
        }

        Ok(())
    }
}

impl CudaServer {
    /// Create a new cuda server.
    pub(crate) fn new(
        ctx: CudaContext,
        mem_props: MemoryDeviceProperties,
        mem_config: MemoryConfiguration,
        mem_alignment: usize,
        device_id: DeviceId,
        utilities: ServerUtilities<Self>,
    ) -> Self {
        let config = CubeClRuntimeConfig::get();
        let max_streams = config.streaming.max_streams;
        let stream_priority = config.streaming.priority;

        ctx.unsafe_set_current().unwrap();

        let comm_stream = crate::compute::stream::create_cuda_stream(stream_priority);

        Self {
            ctx,
            device_id,
            streams: MultiStream::new(
                utilities.logger.clone(),
                CudaStreamBackend::new(
                    mem_props,
                    mem_config,
                    mem_alignment,
                    utilities.logger.clone(),
                    stream_priority,
                ),
                max_streams,
            ),
            utilities: Arc::new(utilities),
            comm_stream,
            communicators: HashMap::default(),
            graphs: HashMap::new(),
        }
    }

    fn command_no_inputs(
        &mut self,
        stream_id: StreamId,
        mode: StreamErrorMode,
    ) -> Result<Command<'_>, ServerError> {
        self.command(stream_id, [].into_iter(), mode)
    }

    fn unsafe_set_current(&self) {
        // TODO: Should check if on the same thread before calling it, since now we don't switch
        // thread except for device memory transfer.
        self.ctx.unsafe_set_current().unwrap();
    }

    fn command<'a>(
        &mut self,
        stream_id: StreamId,
        handles: impl Iterator<Item = &'a Binding>,
        mode: StreamErrorMode,
    ) -> Result<Command<'_>, ServerError> {
        self.unsafe_set_current();

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
        let errors = core::mem::take(&mut stream.current().errors);

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
        kernel: Box<dyn CubeTask<CudaCompiler>>,
        count: CubeCount,
        bindings: KernelArguments,
        mode: ExecutionMode,
        stream_id: StreamId,
        launch_mode: LaunchMode,
    ) -> Result<(), ServerError> {
        let mut kernel_id = kernel.id();
        let logger = self.streams.logger.clone();
        kernel_id.mode(mode);
        let grid_constants = self
            .ctx
            .compilation_options
            .supports_features
            .grid_constants;
        let mut command = self.command(
            stream_id,
            bindings.buffers.iter(),
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        )?;

        let count = match count {
            CubeCount::Static(x, y, z) => (x, y, z),
            // TODO: CUDA doesn't have an exact equivalent of dynamic dispatch. Instead, kernels are free to launch other kernels.
            // One option is to create a dummy kernel with 1 thread that launches the real kernel with the dynamic dispatch settings.
            // For now, just read the dispatch settings from the buffer.
            CubeCount::Dynamic(binding) => {
                let data = future::block_on(command.read_async(vec![CopyDescriptor::new(
                    binding,
                    [3].into(),
                    [1].into(),
                    4,
                )]))?;
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

        let (info_const, info_binding) = if grid_constants {
            let info = &bindings.info;

            let mut handle = Option::None;
            if info.dynamic_metadata_offset < info.data.len() {
                // Only the dynamic tail (shape/stride arrays) becomes a device
                // buffer on this path; scalars and static metadata ride in the
                // kernel's parameter block via `info_const` below.
                let dyn_meta = &info.data[info.dynamic_metadata_offset..];
                handle = Some(info_buffer(&mut command, dyn_meta)?);
            }

            (Some(info.data.as_ptr() as *mut c_void), handle)
        } else {
            let mut handle = Option::None;
            if !bindings.info.data.is_empty() {
                handle = Some(info_buffer(&mut command, &bindings.info.data)?);
            }
            (None, handle)
        };

        let mut resources = bindings
            .buffers
            .into_iter()
            .map(|binding| command.resource(binding).expect("Resource to exist."))
            .collect::<Vec<_>>();

        let mut tensor_maps = Vec::with_capacity(bindings.tensor_maps.len());

        for TensorMapBinding { map, binding } in bindings.tensor_maps.into_iter() {
            let resource = command
                .resource(binding)
                .expect("Tensor map resource exists.");
            let device_ptr = resource.ptr as *mut c_void;

            let mut map_ptr = MaybeUninit::zeroed();

            let shape: Vec<_> = map
                .metadata
                .shape()
                .iter()
                .rev()
                .map(|s| *s as u64)
                .collect();
            let strides: Vec<_> = map
                .metadata
                .strides()
                .iter()
                .rev()
                .skip(1)
                .map(|s| *s as u64 * map.storage_ty.size() as u64)
                .collect();
            let elem_stride: Vec<_> = map.elem_stride.iter().rev().map(|s| *s as u32).collect();

            match &map.format {
                // SAFETY: `map_ptr` is a zeroed `MaybeUninit<CUtensorMap>`. `device_ptr` is a
                // valid device pointer. Shape, strides, tile_size, and elem_stride vectors
                // are constructed from validated metadata and outlive this call.
                TensorMapFormat::Tiled(TiledArgs { tile_size }) => unsafe {
                    let tile_size: Vec<_> =
                        tile_size.iter().rev().copied().map(|s| s as u32).collect();

                    cuTensorMapEncodeTiled(
                        map_ptr.as_mut_ptr(),
                        elem_to_tensor_map_type(map.storage_ty),
                        map.metadata.rank() as u32,
                        device_ptr,
                        shape.as_ptr(),
                        strides.as_ptr(),
                        tile_size.as_ptr(),
                        elem_stride.as_ptr(),
                        interleave_to_cuda(map.interleave),
                        swizzle_to_cuda(map.swizzle),
                        prefetch_to_cuda(map.prefetch),
                        oob_to_cuda(map.oob_fill),
                    )
                    .result()
                    .map_err(|err| {
                        let generic_err =
                            check_tma_generic(&map, device_ptr, &shape, &strides, &elem_stride)
                                .err();
                        let tiled_err = check_tma_tiled(&map, &tile_size).err();
                        generic_err
                            .or(tiled_err)
                            .unwrap_or_else(|| LaunchError::Unknown {
                                reason: format!("{err}"),
                                backtrace: BackTrace::capture(),
                            })
                    })?;
                },
                // SAFETY: Same invariants as `Tiled` above. Additionally, `lower_corner` and
                // `upper_corner` are valid pixel box bounds derived from the tensor map args.
                TensorMapFormat::Im2col(args) => unsafe {
                    let lower_corner: Vec<_> =
                        args.pixel_box_lower_corner.iter().rev().copied().collect();
                    let upper_corner: Vec<_> =
                        args.pixel_box_upper_corner.iter().rev().copied().collect();

                    cuTensorMapEncodeIm2col(
                        map_ptr.as_mut_ptr(),
                        elem_to_tensor_map_type(map.storage_ty),
                        map.metadata.rank() as u32,
                        device_ptr,
                        shape.as_ptr(),
                        strides.as_ptr(),
                        lower_corner.as_ptr(),
                        upper_corner.as_ptr(),
                        args.channels_per_pixel,
                        args.pixels_per_column,
                        elem_stride.as_ptr(),
                        interleave_to_cuda(map.interleave),
                        swizzle_to_cuda(map.swizzle),
                        prefetch_to_cuda(map.prefetch),
                        oob_to_cuda(map.oob_fill),
                    )
                    .result()
                    .map_err(|err| {
                        let generic_err =
                            check_tma_generic(&map, device_ptr, &shape, &strides, &elem_stride)
                                .err();
                        let tiled_err = check_tma_im2col(
                            &map,
                            &lower_corner,
                            &upper_corner,
                            args.channels_per_pixel,
                            args.pixels_per_column,
                        )
                        .err();
                        generic_err
                            .or(tiled_err)
                            .unwrap_or_else(|| LaunchError::Unknown {
                                reason: format!("{err}"),
                                backtrace: BackTrace::capture(),
                            })
                    })?;
                },
                // SAFETY: Same invariants as `Im2col` above. Requires CUDA 12.8+.
                #[cfg(cuda_12080)]
                TensorMapFormat::Im2colWide(args) => unsafe {
                    use cudarc::driver::sys::{
                        CUtensorMapIm2ColWideMode, cuTensorMapEncodeIm2colWide,
                    };
                    cuTensorMapEncodeIm2colWide(
                        map_ptr.as_mut_ptr(),
                        elem_to_tensor_map_type(map.storage_ty),
                        map.metadata.rank() as u32,
                        device_ptr,
                        shape.as_ptr(),
                        strides.as_ptr(),
                        args.pixel_box_lower_corner_width,
                        args.pixel_box_upper_corner_width,
                        args.channels_per_pixel,
                        args.pixels_per_column,
                        elem_stride.as_ptr(),
                        interleave_to_cuda(map.interleave),
                        CUtensorMapIm2ColWideMode::CU_TENSOR_MAP_IM2COL_WIDE_MODE_W,
                        swizzle_to_cuda(map.swizzle),
                        prefetch_to_cuda(map.prefetch),
                        oob_to_cuda(map.oob_fill),
                    )
                    .result()
                    .map_err(|err| {
                        let generic_err =
                            check_tma_generic(&map, device_ptr, &shape, &strides, &elem_stride)
                                .err();
                        generic_err.unwrap_or_else(|| LaunchError::Unknown {
                            reason: format!("{err}"),
                            backtrace: BackTrace::capture(),
                        })
                    })?;
                },
                #[cfg(not(cuda_12080))]
                TensorMapFormat::Im2colWide(_) => {
                    return Err(LaunchError::Unknown {
                        reason: "CUDA version 12.8 required for tensor map format Im2colWide"
                            .into(),
                        backtrace: BackTrace::capture(),
                    }
                    .into());
                }
            };
            // SAFETY: `map_ptr` was fully initialized by one of the `cuTensorMapEncode*`
            // calls above, which all succeeded (errors are propagated before reaching here).
            let binding = unsafe { map_ptr.assume_init() };
            tensor_maps.push(binding);
        }

        resources.extend(
            info_binding
                .into_iter()
                .map(|s| command.resource(s.binding()).expect("Resource to exist")),
        );

        command.kernel(
            kernel_id,
            kernel,
            mode,
            count,
            &tensor_maps,
            &resources,
            info_const,
            logger,
            launch_mode,
        )?;

        Ok(())
    }

    /// Enqueue a graph replay, returning any error to [`replay`](ComputeServer::replay)
    /// to push onto the stream's error queue. Mirrors [`launch_checked`]: the
    /// stream's existing errors are ignored (they surface on the next sync) so a
    /// replay just adds its own on failure.
    ///
    /// [`launch_checked`]: Self::launch_checked
    fn replay_checked(&mut self, graph: GraphId, stream_id: StreamId) -> Result<(), ServerError> {
        // Copy the executable pointer out before borrowing a `command` (which
        // borrows `self`); a raw `CUgraphExec` is `Copy`.
        let exec = self
            .graphs
            .get(&graph)
            .map(|cuda| cuda.exec)
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
        let status = unsafe { cudarc::driver::sys::cuGraphLaunch(exec, stream.sys) };
        cuda_check("cuGraphLaunch", status)
    }

    pub(crate) fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }
}

fn elem_to_tensor_map_type(ty: StorageType) -> CUtensorMapDataType {
    use cudarc::driver::sys::CUtensorMapDataType::*;
    match ty {
        // packed fp4 should be treated as single 4-bit values to simplify indexing/shape handling
        // So a tile of width 16 with fp4 elements is 8 x fp4x2 elements wide.
        #[cfg(cuda_12080)]
        StorageType::Packed(ty, 2) if ty.size_bits() == 4 => CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
        StorageType::Scalar(ElemType::Float(kind)) => match kind {
            // There's no special handling for FP8, so load as u8. `0u8 == 0.0` when reinterpreting.
            FloatKind::E2M1 // single fp4s are padded to a full byte
            | FloatKind::E4M3
            | FloatKind::E5M2
            | FloatKind::UE8M0
            | FloatKind::E2M3
            | FloatKind::E3M2 => CU_TENSOR_MAP_DATA_TYPE_UINT8,
            FloatKind::F16 => CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            FloatKind::BF16 => CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            FloatKind::Flex32 | FloatKind::F32 => CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            FloatKind::TF32 => CU_TENSOR_MAP_DATA_TYPE_TFLOAT32,
            FloatKind::F64 => CU_TENSOR_MAP_DATA_TYPE_FLOAT64,
        },
        StorageType::Scalar(ElemType::Int(kind)) => match kind {
            // UInt is fine because zero bits and size is the same between both
            IntKind::I8 => CU_TENSOR_MAP_DATA_TYPE_UINT8,
            IntKind::I16 => CU_TENSOR_MAP_DATA_TYPE_UINT16,
            IntKind::I32 => CU_TENSOR_MAP_DATA_TYPE_INT32,
            IntKind::I64 => CU_TENSOR_MAP_DATA_TYPE_INT64,
        },
        StorageType::Scalar(ElemType::UInt(kind)) => match kind {
            UIntKind::U8 => CU_TENSOR_MAP_DATA_TYPE_UINT8,
            UIntKind::U16 => CU_TENSOR_MAP_DATA_TYPE_UINT16,
            UIntKind::U32 => CU_TENSOR_MAP_DATA_TYPE_UINT32,
            UIntKind::U64 => CU_TENSOR_MAP_DATA_TYPE_UINT64,
        },
        _ => unimplemented!("Not supported for tensor map type"),
    }
}

fn interleave_to_cuda(interleave: TensorMapInterleave) -> CUtensorMapInterleave {
    use cudarc::driver::sys::CUtensorMapInterleave::*;
    match interleave {
        TensorMapInterleave::None => CU_TENSOR_MAP_INTERLEAVE_NONE,
        TensorMapInterleave::B16 => CU_TENSOR_MAP_INTERLEAVE_16B,
        TensorMapInterleave::B32 => CU_TENSOR_MAP_INTERLEAVE_32B,
    }
}

fn swizzle_to_cuda(swizzle: TensorMapSwizzle) -> CUtensorMapSwizzle {
    use cudarc::driver::sys::CUtensorMapSwizzle::*;
    match swizzle {
        TensorMapSwizzle::None => CU_TENSOR_MAP_SWIZZLE_NONE,
        TensorMapSwizzle::B32 => CU_TENSOR_MAP_SWIZZLE_32B,
        TensorMapSwizzle::B64 => CU_TENSOR_MAP_SWIZZLE_64B,
        TensorMapSwizzle::B128 => CU_TENSOR_MAP_SWIZZLE_128B,
        #[cfg(cuda_12080)]
        TensorMapSwizzle::B128Atom32B => CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B,
        #[cfg(cuda_12080)]
        TensorMapSwizzle::B128Atom32BFlip8B => CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B,
        #[cfg(cuda_12080)]
        TensorMapSwizzle::B128Atom64B => CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B,
        #[cfg(not(cuda_12080))]
        _ => unimplemented!("Swizzle atomicity requires CUDA 12.8 or higher"),
    }
}

fn prefetch_to_cuda(prefetch: TensorMapPrefetch) -> CUtensorMapL2promotion {
    use cudarc::driver::sys::CUtensorMapL2promotion::*;
    match prefetch {
        TensorMapPrefetch::None => CU_TENSOR_MAP_L2_PROMOTION_NONE,
        TensorMapPrefetch::B64 => CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
        TensorMapPrefetch::B128 => CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        TensorMapPrefetch::B256 => CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
    }
}

fn oob_to_cuda(fill: OobFill) -> CUtensorMapFloatOOBfill {
    use cudarc::driver::sys::CUtensorMapFloatOOBfill::*;
    match fill {
        OobFill::Zero => CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        OobFill::NaN => CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA,
    }
}

macro_rules! launch_check {
    ($assertion: expr, $($arg:tt)+) => {
        if $assertion {
            Ok(())
        } else {
            Err(LaunchError::Unknown {
                reason: format!($($arg)*),
                backtrace: BackTrace::capture(),
            })
        }
    };
}

fn check_tma_generic(
    map: &TensorMapMeta,
    device_ptr: *mut c_void,
    shape: &[u64],
    strides: &[u64],
    elem_strides: &[u32],
) -> Result<(), LaunchError> {
    // globalAddress invariants
    launch_check!(
        (device_ptr as usize).is_multiple_of(16),
        "Tensor pointer must be 16 byte aligned"
    )?;
    if !matches!(map.interleave, TensorMapInterleave::None) {
        launch_check!(
            (device_ptr as usize).is_multiple_of(32),
            "Tensor pointer must be 32 byte aligned"
        )?;
    }

    // tensorRank invariants
    launch_check!(
        (1..=5).contains(&map.metadata.rank()),
        "Rank must be between 1 and 5"
    )?;
    launch_check!(
        matches!(map.interleave, TensorMapInterleave::None) || map.metadata.rank() >= 3,
        "When interleave is enabled, rank must be >= 3"
    )?;

    // globalDim invariants
    launch_check!(
        shape.iter().all(|it| *it <= u32::MAX as u64),
        "Shape must be <= u32::MAX"
    )?;
    #[cfg(cuda_12080)]
    if matches!(map.storage_ty, StorageType::Packed(ty, 2) if ty.size_bits() == 4) {
        launch_check!(
            shape[0].is_multiple_of(2),
            "Packed tensor map must have multiple of 2 for the innermost dimension"
        )?;
    }

    // globalStrides invariants
    launch_check!(
        strides.iter().all(|it| it.is_multiple_of(16)),
        "Strides must be 16 byte aligned"
    )?;
    if matches!(map.interleave, TensorMapInterleave::B32) {
        launch_check!(
            strides.iter().all(|it| it.is_multiple_of(32)),
            "Strides must be 32 byte aligned when interleave is B32"
        )?;
    }

    // elementStrides invariants
    launch_check!(
        elem_strides.iter().all(|it| *it > 0 && *it <= 8),
        "Element strides must be non-zero and <= 8"
    )?;
    if matches!(map.interleave, TensorMapInterleave::None) {
        launch_check!(
            elem_strides[0] == 1,
            "Innermost element stride is ignored without interleaving"
        )?;
    }

    // oobFill invariants
    if matches!(map.oob_fill, OobFill::NaN) {
        launch_check!(
            map.storage_ty.is_float(),
            "NaN fill is only supported for float types"
        )?;
    }

    Ok(())
}

fn check_tma_tiled(map: &TensorMapMeta, tile_size: &[u32]) -> Result<(), LaunchError> {
    launch_check!(
        tile_size.len() == map.metadata.rank(),
        "Tile shape should match rank"
    )?;
    launch_check!(
        tile_size.iter().all(|it| *it > 0 && *it <= 256),
        "Tile shape must be non-zero and <= 256"
    )?;
    let tile_size_0_bytes = tile_size[0] as usize * map.storage_ty.size();
    if matches!(map.interleave, TensorMapInterleave::None) {
        let max_tile_bytes = match map.swizzle {
            TensorMapSwizzle::None => usize::MAX,
            TensorMapSwizzle::B32 => 32,
            TensorMapSwizzle::B64 => 64,
            TensorMapSwizzle::B128
            | TensorMapSwizzle::B128Atom32B
            | TensorMapSwizzle::B128Atom32BFlip8B
            | TensorMapSwizzle::B128Atom64B => 128,
        };
        launch_check!(
            tile_size_0_bytes <= max_tile_bytes,
            "Innermost tile dim must be <= swizzle size"
        )?;
    }
    if matches!(map.interleave, TensorMapInterleave::B32) {
        launch_check!(
            map.swizzle == TensorMapSwizzle::B32,
            "If interleave is B32, swizzle must be B32"
        )?;
    }

    Ok(())
}

fn check_tma_im2col(
    map: &TensorMapMeta,
    lower_corner: &[i32],
    upper_corner: &[i32],
    channels_per_pixel: u32,
    pixels_per_column: u32,
) -> Result<(), LaunchError> {
    launch_check!(
        lower_corner.len() == map.metadata.rank() - 2,
        "Lower corner must be rank - 2 elements"
    )?;
    launch_check!(
        upper_corner.len() == map.metadata.rank() - 2,
        "Upper corner must be rank - 2 elements"
    )?;

    launch_check!(
        map.metadata.rank() >= 3 && map.metadata.rank() <= 5,
        "im2col requires rank to be between 3 and 5"
    )?;

    let (range_lower, range_upper) = match map.metadata.rank() {
        3 => (-32768, 32767),
        4 => (-128, 127),
        5 => (-16, 15),
        _ => unreachable!(),
    };
    launch_check!(
        lower_corner
            .iter()
            .all(|it| *it >= range_lower && *it <= range_upper),
        "Lower corner must be in range [{range_lower}, {range_upper}] for {}D im2col",
        map.metadata.rank()
    )?;
    launch_check!(
        upper_corner
            .iter()
            .all(|it| *it >= range_lower && *it <= range_upper),
        "Upper corner must be in range [{range_lower}, {range_upper}] for {}D im2col",
        map.metadata.rank()
    )?;

    launch_check!(
        channels_per_pixel <= 256,
        "Channels per pixel must be <= 256"
    )?;
    launch_check!(
        pixels_per_column <= 1024,
        "Pixels per column must be <= 1024"
    )?;

    Ok(())
}
