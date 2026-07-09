use super::storage::gpu::{GpuResource, GpuStorage};
use crate::{
    compute::{
        command::Command,
        context::HipContext,
        fence::Fence,
        stream::{HipStreamBackend, StreamCaptureState},
    },
    runtime::HipCompiler,
};
use cubecl_common::{bytes::Bytes, future::DynFut, profile::ProfileDuration, stream_id::StreamId};
use cubecl_core::{
    MemoryConfiguration,
    backtrace::BackTrace,
    future,
    ir::MemoryDeviceProperties,
    prelude::*,
    server::{
        Binding, CopyDescriptor, KernelArguments, ProfileError, ProfilingToken,
        ServerCommunication, ServerError, ServerUtilities, StreamErrorMode,
    },
};
use cubecl_runtime::{
    allocator::PitchedMemoryLayoutPolicy,
    compiler::CubeTask,
    config::{CubeClRuntimeConfig, RuntimeConfig},
    id::GraphId,
    logging::ServerLogger,
    memory_management::{ManagedMemoryHandle, MemoryAllocationMode, MemoryUsage},
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

/// Build a [`ServerError`] for a graph-capture call issued in the wrong state
/// (e.g. `begin_capture` without `graph_prepare`, or a second overlapping
/// capture on the same stream).
fn graph_state_error(reason: impl Into<String>) -> ServerError {
    ServerError::Generic {
        reason: reason.into(),
        backtrace: BackTrace::capture(),
    }
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

        let reserved = command.reserve(size).unwrap();
        command.bind(reserved, memory);
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
    ) {
        if let Err(err) = self.launch_checked(kernel, count, bindings, mode, stream_id) {
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
        // the capture window then reuses those slices with no `hipMalloc`
        // (which would be illegal mid-capture, HIP status 901). `end_capture`
        // pins everything the window added on the graph.
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
        // the window reuse slices with no illegal mid-capture `hipMalloc`.
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
        // Build the graph inside a scope so the `command` borrow of `self` ends
        // before we register the graph in `self.graphs`.
        let hip_graph = {
            let mut command = self.command_no_inputs(
                stream_id,
                StreamErrorMode {
                    ignore: false,
                    flush: true,
                },
            )?;
            let stream = command.streams.current();
            // Only a recording stream can be ended; reject a stray `end_capture`
            // (nothing prepared/started, or the capture already ended) instead of
            // calling `hipStreamEndCapture` on a stream that never began one.
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
                let mut graph: cubecl_hip_sys::hipGraph_t = std::ptr::null_mut();
                hip_check(
                    "hipStreamEndCapture",
                    cubecl_hip_sys::hipStreamEndCapture(stream.sys, &mut graph),
                )
                .and_then(|_| {
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
            // The capture is over even if it failed to instantiate: re-enable the
            // deferred fenced flushes and restore the allocation mode, so an error
            // here doesn't leave the stream stuck in capture/persistent state.
            stream.capturing = StreamCaptureState::NoCapture;
            // Pin every buffer the graph touched so the pool never reuses that
            // memory for the graph's lifetime — both the GPU slices and the pinned
            // staging slices the recorded info copies still read from on replay.
            // On failure the handles drop with `exec?` below, unpinning them.
            let mut retained = stream.memory_management_gpu.capture_end();
            retained.extend(stream.memory_management_cpu.capture_end());
            // Reclaim the buffers dropped during the capture window, whose fenced
            // flushes were deferred while `capturing` was set. Flush twice: the
            // queue is a double buffer, one flush only rotates the current batch.
            let sys = stream.sys;
            stream.drop_queue.flush(|| Fence::new(sys));
            stream.drop_queue.flush(|| Fence::new(sys));
            HipGraph {
                exec: exec?,
                _retained: retained,
            }
        };
        let id = GraphId::new();
        self.graphs.insert(id, hip_graph);
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
        // Sync the stream, then drop the graph — `HipGraph::drop` destroys the
        // executable and unpins the buffers it retained. No-op for an unknown id
        // (e.g. a double release).
        if self.graphs.contains_key(&graph) {
            let _ = cubecl_common::future::block_on(self.sync(stream_id));
            self.graphs.remove(&graph);
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
        cubecl_common::future::block_on(self.sync(stream_id))?;
        Ok(self.ctx.timestamps.start())
    }

    fn end_profile(
        &mut self,
        stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        if let Err(err) = cubecl_common::future::block_on(self.sync(stream_id)) {
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
        handles: impl Iterator<Item = &'a Binding>,
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
        kernel: Box<dyn CubeTask<HipCompiler>>,
        count: CubeCount,
        bindings: KernelArguments,
        mode: ExecutionMode,
        stream_id: StreamId,
    ) -> Result<(), ServerError> {
        let mut kernel_id = kernel.id();
        let logger = self.streams.logger.clone();
        kernel_id.mode(mode);
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

        let KernelArguments {
            buffers,
            info,
            tensor_maps,
        } = bindings;

        debug_assert!(tensor_maps.is_empty(), "Can't use tensor maps on HIP");

        // Reuse a cached info buffer when this kernel has already run with these
        // exact shapes/scalars. The info is read-only metadata (no tensor
        // pointers), so sharing it across launches is sound — and it means a
        // stable-shape decode allocates and copies no info inside a capture
        // window (all launches hit warm buffers).
        //
        // The cache's policy makes every decision (see `MetadataInfoCache`), and
        // the capture lifecycle drives its mode so that during capture every
        // buffer is cached and none is evicted. We ask the policy first and only
        // touch the cache when it says to — otherwise we just build the buffer,
        // never cloning a handle we wouldn't keep.
        let key = (kernel_id.clone(), info.data.clone());
        let size = core::mem::size_of_val(info.data.as_slice());
        let cache_mode = command.streams.current().capturing.cache_mode();
        command.streams.current().info_cache.mode(cache_mode);

        let info_handle = if command.streams.current().info_cache.should_cache(size) {
            match command.streams.current().info_cache.get(&key) {
                Some(handle) => handle,
                None => {
                    let handle = command
                        .create_with_data(bytemuck::cast_slice(&info.data))
                        .unwrap();
                    command
                        .streams
                        .current()
                        .info_cache
                        .insert(key, handle.clone());
                    handle
                }
            }
        } else {
            command
                .create_with_data(bytemuck::cast_slice(&info.data))
                .unwrap()
        };

        let mut resources: Vec<_> = buffers
            .into_iter()
            .map(|b| command.resource(b).expect("Resource to exist."))
            .collect();

        resources.push(
            command
                .resource(info_handle.binding())
                .expect("Resource to exist."),
        );

        command.kernel(kernel_id, kernel, mode, count, &resources, logger)?;

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
        let exec = self
            .graphs
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
