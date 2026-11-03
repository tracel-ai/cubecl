use super::storage::gpu::{GpuResource, GpuStorage};
use crate::compute::driver::Cuda;
use crate::compute::modules::CudaCompiledKernel;
use crate::compute::{Captures, Command, Window, context::CudaContext, stream::CudaStreamBackend};
use cubecl_common::device::Device;
use cubecl_common::{bytes::Bytes, profile::ProfileDuration};
use cubecl_core::server::{DeviceCaptures, ServerStorage};
use cubecl_core::{
    MemoryConfiguration,
    device::DeviceId,
    ir::{ElemType, FloatKind, IntKind, MemoryDeviceProperties, UIntKind},
    prelude::*,
    server::{
        BufferBinding, ComputeWait, CopyDescriptor, Handle, KernelArguments, KernelResource,
        LaunchError, PeerBinding, ProfileError, ProfilingToken, Reduction, ServerCommunication,
        ServerError, ServerUtilities, TensorMapBinding, TensorMapMeta,
    },
    zspace::SmallVec,
};
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::future::{self, DynFut};
use cubecl_environment::stream::StreamId;
use cubecl_server::command::{CollectiveServer, Collectives, PeerLaunch, Refused};
use cubecl_server::compiler::ArtifactId;
use cubecl_server::device::CudaDevice;
use cubecl_server::{
    config::{CubeClRuntimeConfig, RuntimeConfig},
    execution::LaunchMode,
    id::GraphId,
    kernel::CubeKernel,
    logging::ServerLogger,
    memory_management::{ManagedMemoryHandle, MemoryAllocationMode, StreamMemoryReport},
    server::Server,
    storage::{ComputeStorage, ManagedResource},
    stream::{
        ExecuteScope, FailureStore, MultiStream, ScopedOutcome, StreamCapture, WriteScoped,
        failed_writing,
    },
};
use cudarc::driver::sys::{
    CUstream, CUtensorMap, CUtensorMapDataType, CUtensorMapFloatOOBfill, CUtensorMapInterleave,
    CUtensorMapL2promotion, CUtensorMapSwizzle, cuTensorMapEncodeIm2col, cuTensorMapEncodeTiled,
};
use std::collections::HashSet;
use std::{ffi::c_void, sync::Arc};

/// Stage `words` into a device buffer, reusing a cached one when a launch has
/// already staged these exact info words. The info is read-only metadata (no
/// tensor pointers), so sharing it across launches — even of different kernels
/// — is sound, and it means a stable-shape decode allocates and copies no info
/// inside a capture window (all launches hit warm buffers, so the captured
/// graph gains no memcpy nodes for them).
///
/// The cache's policy makes every decision (see
/// [`MetadataInfoCache`](cubecl_server::metadata_cache::MetadataInfoCache)),
/// and the capture lifecycle drives its mode so that during capture every
/// buffer is cached and none is evicted. We ask the policy first and only touch
/// the cache when it says to — otherwise we just build the buffer, never
/// cloning a key we wouldn't keep. The buffer's bytes always equal the key
/// bytes, so a hit is byte-identical to what the miss path would have built.
fn info_buffer(command: &mut Command<'_>, words: &[u64]) -> Result<Handle, ServerError> {
    let size = core::mem::size_of_val(words);
    let cache_mode = command.stream().capturing.cache_mode();
    command.stream().info_cache.mode(cache_mode);

    if !command.stream().info_cache.should_cache(size) {
        return Ok(command.create_with_data(bytemuck::cast_slice(words))?);
    }
    // Look up by the borrowed words — a hit clones nothing. On a miss we build
    // the buffer and clone the words into the cache as the key.
    if let Some(handle) = command.stream().info_cache.get(words) {
        return Ok(handle);
    }
    let handle = command.create_with_data(bytemuck::cast_slice(words))?;
    command
        .stream()
        .info_cache
        .insert(words.to_vec(), handle.clone());
    Ok(handle)
}

#[derive(Debug)]
pub struct CudaServer {
    ctx: CudaContext,
    device_id: DeviceId,
    /// The devices whose memory kernels here can address: peer access is
    /// enabled once per pair, the first time a kernel binds that device's
    /// memory.
    peers: HashSet<DeviceId>,
    streams: MultiStream<CudaStreamBackend>,
    utilities: Arc<ServerUtilities>,
    /// The groups this device has joined — see [`Collectives`].
    collectives: Collectives<Cuda>,
    /// Captured graphs owned by this server, keyed by the [`GraphId`] handed to
    /// the client. `end_capture` inserts, `replay` looks up, `graph_destroy`
    /// removes (dropping the [`CudaGraph`] destroys its executable and unpins the
    /// buffers it retained). Referencing graphs by id keeps the raw
    /// `CUgraphExec` inside the server, never boxed across the actor boundary.
    graphs: Captures,
}

// SAFETY: `CudaServer` is only accessed from one thread at a time via the `DeviceHandle`,
// which serializes all server access. The CUDA context, streams, and NCCL communicators
// it manages are never shared across threads without synchronization.
unsafe impl Send for CudaServer {}

impl Server for CudaServer {
    fn logger(&self) -> Arc<ServerLogger> {
        self.streams.logger.clone()
    }

    fn staging(&mut self, sizes: &[usize], stream_id: StreamId) -> Result<Vec<Bytes>, ServerError> {
        let mut command = self.command_no_inputs(stream_id)?;

        Ok(sizes
            .iter()
            .map(|size| command.reserve_cpu(*size, None))
            .collect())
    }

    fn utilities(&self) -> Arc<ServerUtilities> {
        self.utilities.clone()
    }

    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        // The bytes are only as good as the work that wrote them.
        if let Err(err) = self
            .streams
            .ensure_written(descriptors.iter().map(|d| &d.handle))
        {
            return Box::pin(async move { Err(err) });
        }

        let mut command = match self.command(stream_id, descriptors.iter().map(|d| &d.handle)) {
            Ok(command) => command,
            Err(err) => return Box::pin(async move { Err(err) }),
        };
        Box::pin(command.read_async(descriptors))
    }

    fn initialize_memory(
        &mut self,
        memory: ManagedMemoryHandle,
        size: u64,
        stream_id: StreamId,
    ) -> Result<(), ServerError> {
        let result = match self.command_no_inputs(stream_id) {
            Ok(mut command) => command
                .initialize_memory(memory.clone(), size)
                .map_err(ServerError::from),
            Err(error) => Err(error),
        };
        // No stream to allocate on, or no memory to allocate: the buffer carries the
        // error instead.
        if let Err(error) = &result {
            self.streams.fail_unallocated(&memory, error.clone());
        }
        result
    }

    fn write(&mut self, descriptors: Vec<(CopyDescriptor, Bytes)>, stream_id: StreamId) {
        // Each copy runs in its own scope over its destination: the copy fills
        // it on success, which is what releases an earlier failure's hold on
        // it — a buffer a launch left stale is recovered by writing it from
        // the host just as much as by relaunching into it — and leaves it as
        // it was on failure, which is what a later read of it has to fail on.
        //
        // Every descriptor is attempted, however the one before it went. A
        // copy that stops early leaves the destinations it never reached
        // holding whatever was there before and carrying no failure to say so,
        // which is the one outcome this whole design exists to prevent — and
        // the failure of one copy says nothing about the next, which may name
        // a different buffer on a different stream.
        for (descriptor, data) in descriptors {
            let mut written = self.write_set();
            written.push(descriptor.handle.clone());
            ExecuteScope::over(self, stream_id, written).execute(|server| {
                let mut command = server.command(stream_id, [&descriptor.handle].into_iter())?;
                command.write_to_gpu(descriptor, data).map_err(Into::into)
            });
        }
    }

    unsafe fn launch(
        &mut self,
        kernel: Box<dyn CubeKernel>,
        count: CubeCount,
        bindings: KernelArguments,
        stream_id: StreamId,
        launch_mode: LaunchMode,
    ) {
        if launch_mode == LaunchMode::Queue {
            self.ctx.queue_kernel(kernel);
            return;
        }
        let id = ArtifactId {
            kernel: kernel.id(),
            variant: (),
        };
        let Some(loaded) = self.load_or_fail(kernel, &id, &bindings, stream_id, launch_mode) else {
            return;
        };
        // A discarded launch stops right here, after compilation and before anything
        // that touches a buffer: resolving resources, building tensor maps,
        // uploading metadata or reading a dynamic cube count would materialize
        // memory the run exists to leave unmapped.
        if launch_mode.discards_launch() {
            return;
        }
        let kernel_id = id.kernel;
        let io = loaded.io.as_deref();
        // What the launch does with memory of other devices, settled with
        // their owners however the launch ends.
        let peers = PeerLaunch::new(&bindings, io);

        // The count resolves before the scope opens, because entering the
        // scope replaces whatever claim the outputs carry — and a count that
        // resolves to zero enqueues nothing, so those claims must be left
        // exactly as they were, which no exit can restore once entry took
        // them.
        let count = match self.resolve_cube_count(count, stream_id) {
            Ok(count) => count,
            Err(err) => {
                // The launch cannot run, so its outputs take the failure
                // exactly as a failed launch's would: a tainted or unreadable
                // count buffer travels to everything downstream of it.
                let mut written = self.write_set();
                written.extend(bindings.buffers_written(io).cloned());
                peers.failed(&err);
                failed_writing(self, stream_id, written, err);
                return;
            }
        };
        // Zero threads: the driver rejects a zero grid dim, and a launch of
        // zero threads writes nothing — no scope opens, so a claim an earlier
        // failure holds on the outputs stays exactly where it was.
        if count.0 == 0 || count.1 == 0 || count.2 == 0 {
            return;
        }

        // The scope claims what the launch writes until the body proves the
        // work enqueued, so a failure — or a panic — anywhere in it leaves a
        // read of those buffers failing on the error rather than copying
        // bytes nothing wrote. An input that already carries a failure skips
        // the launch instead, and the scope settles that too.
        let mut written = self.write_set();
        written.extend(bindings.buffers_written(io).cloned());
        // A peer input that carries a failure stops the launch as a failed
        // input of this device would: nothing runs on its stale bytes.
        if let Some(error) = peers.input_failure() {
            peers.failed(&error);
            failed_writing(self, stream_id, written, error);
            return;
        }
        let outcome = ExecuteScope::launching(
            self,
            kernel_id.clone(),
            stream_id,
            bindings.buffers_read(io),
            written,
        )
        .execute(|server| server.launch_checked(kernel_id, &loaded, count, bindings, stream_id));
        match outcome {
            ScopedOutcome::Executed(()) => peers.enqueued(),
            ScopedOutcome::Failed(error) | ScopedOutcome::Skipped(error) => peers.failed(&error),
        }
    }

    fn check(
        &mut self,
        handles: Vec<BufferBinding>,
        _stream_id: StreamId,
    ) -> Result<(), ServerError> {
        self.streams.ensure_written(handles.iter())
    }

    fn compile_queued(&mut self) {
        self.ctx.compile_queued(&self.streams.logger);
    }

    fn flush(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        // A flush reports nothing: a failure lives on the buffers the work
        // left unwritten, and a read of one of them is what surfaces it.
        let mut command = self.command_no_inputs(stream_id)?;
        command.flush_drops();
        command.stream().memory_management_gpu.storage().flush();

        Ok(())
    }

    fn graph_prepare(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        let mut command = self.command_no_inputs(stream_id)?;
        Window::on(&mut command).prepare(stream_id)
    }

    fn begin_capture(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        let mut command = self.command_no_inputs(stream_id)?;
        Window::on(&mut command).begin()
    }

    fn end_capture(&mut self, stream_id: StreamId) -> Result<GraphId, ServerError> {
        let id = GraphId::new();
        let instantiated = {
            let mut command = self.command_no_inputs(stream_id)?;
            Window::on(&mut command).instantiate(stream_id, id)
        };
        match instantiated {
            Ok(graph) => {
                self.graphs.insert(id, graph);
                Ok(id)
            }
            // No graph is handed back, so the recorded launches never run:
            // every buffer they were given is left as it was. The caller gets
            // the error below; the taint is what makes a read of one of those
            // buffers fail on some other stream, which heard nothing.
            Err(Refused { error, written }) => {
                self.streams.taint(error.clone(), written.iter());
                Err(error)
            }
        }
    }

    fn replay(&mut self, graph: GraphId, stream_id: StreamId) -> Result<(), ServerError> {
        // A replay writes the buffers its recorded launches were given, so it
        // takes the same scope over that write set and settles it: a failed
        // enqueue leaves them carrying the failure, and the next replay that
        // lands releases the claim. Without the settle one transient failure
        // would leave the graph's buffers unreadable forever — the graph
        // retains their handles, so none of the shedding paths can ever fire
        // for them, and the graph itself is the only thing that writes them.
        let mut written = self.write_set();
        self.graphs.extend_written(graph, &mut written);
        ExecuteScope::over(self, stream_id, written)
            .execute(|server| {
                let mut streams = server.streams.resolve(stream_id, [].into_iter())?;
                server.graphs.replay(graph, streams.current())
            })
            .into_result()
    }

    fn graph_destroy(&mut self, graph: GraphId, stream_id: StreamId) {
        // No-op for an unknown id (e.g. a double release), and nothing to sync
        // for either.
        if !self.graphs.contains(graph) {
            return;
        }
        // What this graph's replays write, taken before the graph goes: if the
        // wait below fails, these are the buffers a replay may not have
        // finished writing, and the failure belongs to them.
        let mut written = Vec::new();
        self.graphs.extend_written(graph, &mut written);

        // Wait for in-flight replays before dropping the executable: `replay`
        // returns at enqueue time, so one may still be running against it. A
        // failed wait means no replay is still running, so destroying is safe.
        let synced = cubecl_environment::future::block_on(self.sync(Vec::new(), stream_id));
        let stream = self.streams.try_stream_mut(&stream_id);
        self.graphs.destroy(graph, stream);
        if let Err(err) = synced {
            // Claimed rather than reported at large: work on this stream that
            // shares no buffer with the graph has nothing to do with this and
            // must not be failed for it.
            //
            // The wait was over the whole stream and not only this graph's
            // replays, so anything else in flight on it is equally unfinished
            // and goes unclaimed. That is deliberate: whatever else was
            // enqueued reports through its own read, and widening this to the
            // stream is exactly the contamination the claim exists to avoid.
            failed_writing(self, stream_id, written, err);
        }
    }

    fn sync(
        &mut self,
        handles: Vec<BufferBinding>,
        stream_id: StreamId,
    ) -> DynFut<Result<(), ServerError>> {
        // The claim check a read would have made, without the read; claims
        // are set at enqueue time, so they are already in place. A fault the
        // barrier itself reveals comes back through the fence below.
        if let Err(err) = self.streams.ensure_written(handles.iter()) {
            return Box::pin(async move { Err(err) });
        }
        match self.command_no_inputs(stream_id) {
            Ok(mut command) => command.sync(),
            Err(err) => Box::pin(async move { Err(err) }),
        }
    }

    fn start_profile(&mut self, stream_id: StreamId) -> Result<ProfilingToken, ServerError> {
        // No drain: the window opens where the stream already is, and the
        // device stamps it there. See [`EventProfiler`].
        let sys = self.profiled_stream(stream_id, "start_profile")?;
        self.ctx.profiler.start(sys)
    }

    fn start_profile_at_launch(
        &mut self,
        stream_id: StreamId,
    ) -> Result<ProfilingToken, ServerError> {
        // Stamped by the launch itself, right before `cuLaunchKernel`
        // (`CudaContext::execute_task`).
        let sys = self.profiled_stream(stream_id, "start_profile_at_launch")?;
        self.ctx.profiler.start_at_launch(sys)
    }

    fn end_profile(
        &mut self,
        stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        let sys = match self.profiled_stream(stream_id, "end_profile") {
            Ok(sys) => sys,
            Err(err) => {
                // The window cannot be closed on the device, so there is
                // nothing to read back — drop it and report, rather than leave
                // a token open that nothing will ever close.
                self.ctx.profiler.abandon(token);
                return Err(ProfileError::from(&err));
            }
        };
        self.ctx.profiler.stop(sys, token)
    }

    fn abandon_profile(&mut self, _stream_id: StreamId, token: ProfilingToken) {
        // Nothing to record on the device: dropping the window returns its
        // start event to the pool.
        self.ctx.profiler.abandon(token);
    }

    fn memory_report(&mut self, stream_id: StreamId) -> Option<StreamMemoryReport> {
        // A stream that cannot be created has no memory to report.
        self.streams.try_stream_mut(&stream_id)?;
        // The stream exists, so resolving it creates nothing and cannot fail.
        self.command_no_inputs(stream_id)
            .ok()
            .map(|mut command| command.memory_report())
    }

    fn stream_ids(&self) -> Vec<StreamId> {
        self.streams.stream_ids().collect()
    }

    fn memory_cleanup(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        let mut command = self.command_no_inputs(stream_id)?;
        command.memory_cleanup()
    }

    fn allocation_mode(&mut self, mode: MemoryAllocationMode, stream_id: StreamId) {
        // A failed stream creation is reported by the next command on it.
        if let Ok(mut command) = self.command_no_inputs(stream_id) {
            command.allocation_mode(mode);
        }
    }
}

impl ServerCommunication for CudaServer {
    fn comm_init(&mut self, device_ids: Vec<DeviceId>) -> Result<(), ServerError> {
        CollectiveServer::comm_init(self, device_ids)
    }

    fn all_reduce(
        &mut self,
        src: BufferBinding,
        dst: BufferBinding,
        reduction: Reduction,
        stream_id: StreamId,
        device_ids: Vec<DeviceId>,
        wait: ComputeWait,
    ) -> Result<(), ServerError> {
        CollectiveServer::all_reduce(self, src, dst, reduction, stream_id, device_ids, wait)
    }

    fn all_gather(
        &mut self,
        src: BufferBinding,
        dst: BufferBinding,
        stream_id: StreamId,
        device_ids: Vec<DeviceId>,
    ) -> Result<(), ServerError> {
        CollectiveServer::all_gather(self, src, dst, stream_id, device_ids)
    }

    fn reduce_scatter(
        &mut self,
        src: BufferBinding,
        dst: BufferBinding,
        reduction: Reduction,
        stream_id: StreamId,
        device_ids: Vec<DeviceId>,
    ) -> Result<(), ServerError> {
        CollectiveServer::reduce_scatter(self, src, dst, reduction, stream_id, device_ids)
    }

    fn broadcast(
        &mut self,
        src: Option<BufferBinding>,
        dst: BufferBinding,
        stream_id: StreamId,
        root: DeviceId,
        device_ids: Vec<DeviceId>,
    ) -> Result<(), ServerError> {
        CollectiveServer::broadcast(self, src, dst, stream_id, root, device_ids)
    }

    fn all_to_all(
        &mut self,
        src: BufferBinding,
        dst: BufferBinding,
        stream_id: StreamId,
        device_ids: Vec<DeviceId>,
    ) -> Result<(), ServerError> {
        CollectiveServer::all_to_all(self, src, dst, stream_id, device_ids)
    }

    fn sync_collective(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        CollectiveServer::sync_collective(self, stream_id)
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(desc)))]
    fn send(
        &mut self,
        desc: CopyDescriptor,
        stream_id: StreamId,
        device_id_dst: DeviceId,
    ) -> Result<(), ServerError> {
        CollectiveServer::send(self, desc, stream_id, device_id_dst)
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace"))]
    fn recv(
        &mut self,
        handle: Handle,
        stream_id: StreamId,
        device_id_src: DeviceId,
    ) -> Result<(), ServerError> {
        CollectiveServer::recv(self, handle, stream_id, device_id_src)
    }

    fn can_access_peer(&mut self, peer: DeviceId) -> bool {
        use cudarc::driver::sys::CUdevice_P2PAttribute as Attribute;
        let (Ok(own), Ok(other)) = (cuda_device(self.device_id), cuda_device(peer)) else {
            return false;
        };
        // Kernels on two devices hand data off through system-scope atomics
        // on one's memory, so plain loads and stores are not enough: over a
        // PCIe link without atomics, the producer's flag updates can be lost.
        let attribute = |attribute| {
            let mut value = 0;
            // SAFETY: `value` is a valid stack variable the driver writes through.
            let status = unsafe {
                cudarc::driver::sys::cuDeviceGetP2PAttribute(&mut value, attribute, own, other)
            };
            status.result().is_ok() && value != 0
        };
        attribute(Attribute::CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED)
            && attribute(Attribute::CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED)
    }

    fn peer_address(&mut self, binding: BufferBinding) -> Result<u64, ServerError> {
        // The same claim check a read makes: a peer reads these bytes with
        // nothing ordering it behind a failed write here.
        let stream = binding.stream;
        Ok(self.get_resource(binding, stream)?.resource().ptr)
    }

    fn peer_wrote(&mut self, written: Vec<BufferBinding>, error: Option<ServerError>) {
        let Some(stream) = written.first().map(|binding| binding.stream) else {
            return;
        };
        match error {
            Some(error) => failed_writing(self, stream, written, error),
            None => self.streams.written(written.iter()),
        }
    }
}

/// The driver's handle on the device `device_id` names.
fn cuda_device(device_id: DeviceId) -> Result<cudarc::driver::sys::CUdevice, ServerError> {
    let index = CudaDevice::from_id(device_id).index as i32;
    cudarc::driver::result::device::get(index).map_err(|err| ServerError::Generic {
        reason: format!("cuDeviceGet({index}) failed: {err}"),
        backtrace: BackTrace::capture(),
    })
}

impl CudaServer {
    /// Let kernels on this device address the memory of `peer`, once.
    ///
    /// Unified addressing maps every allocation of the peer, present and
    /// future, into this context, so the address the peer resolved is the one
    /// a kernel here dereferences.
    fn enable_peer(&mut self, peer: DeviceId) -> Result<(), ServerError> {
        if self.peers.contains(&peer) {
            return Ok(());
        }
        let device = cuda_device(peer)?;
        // Access is granted to the current context, so this device's is made
        // current first rather than trusting whatever the thread last set.
        self.ctx
            .unsafe_set_current()
            .map_err(|err| ServerError::Generic {
                reason: format!(
                    "making the context of {:?} current failed: {err}",
                    self.device_id
                ),
                backtrace: BackTrace::capture(),
            })?;
        // SAFETY: the peer's primary context is retained for the call and
        // released after: its own server holds it for as long as it runs, and
        // the access lives as long as both contexts do.
        let status = unsafe {
            let peer_ctx = cudarc::driver::result::primary_ctx::retain(device).map_err(|err| {
                ServerError::Generic {
                    reason: format!("retaining the context of {peer:?} failed: {err}"),
                    backtrace: BackTrace::capture(),
                }
            })?;
            let status = cudarc::driver::sys::cuCtxEnablePeerAccess(peer_ctx, 0);
            // Releasing what was retained above cannot fail on a live device.
            let _ = cudarc::driver::result::primary_ctx::release(device);
            status
        };
        if status != cudarc::driver::sys::CUresult::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED {
            status.result().map_err(|err| ServerError::Generic {
                reason: format!("cuCtxEnablePeerAccess to {peer:?} failed: {err}"),
                backtrace: BackTrace::capture(),
            })?;
        }
        self.peers.insert(peer);
        Ok(())
    }
}

impl CollectiveServer for CudaServer {
    type Driver = Cuda;

    fn device_id(&self) -> DeviceId {
        self.device_id
    }

    fn collectives(&mut self) -> &mut Collectives<Cuda> {
        &mut self.collectives
    }

    fn streams(&mut self) -> &mut MultiStream<CudaStreamBackend> {
        &mut self.streams
    }

    fn command<'a>(
        &mut self,
        stream_id: StreamId,
        handles: impl Iterator<Item = &'a BufferBinding>,
    ) -> Result<Command<'_>, ServerError> {
        CudaServer::command(self, stream_id, handles)
    }

    fn comm_stream(&self) -> CUstream {
        self.ctx.comm_stream
    }

    fn transfer_stream(&self) -> CUstream {
        self.ctx.transfer_stream
    }

    fn connect_element(&self) -> &GpuResource {
        &self.ctx.connect_element
    }
}

impl WriteScoped for CudaServer {
    type Streams = MultiStream<CudaStreamBackend>;

    fn write_streams(&mut self) -> &mut Self::Streams {
        &mut self.streams
    }

    fn on_failure(&mut self, _stream: StreamId, error: &ServerError) {
        self.profile_failure(error);
    }

    fn capturing(&mut self, stream: StreamId) -> Option<&mut StreamCapture> {
        self.streams
            .try_stream_mut(&stream)
            .map(|stream| &mut stream.capturing)
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
        utilities: ServerUtilities,
        captures: DeviceCaptures,
    ) -> Self {
        let config = CubeClRuntimeConfig::get();
        let max_streams = config.streaming.max_streams;
        let stream_priority = config.streaming.priority;

        ctx.unsafe_set_current().unwrap();

        Self {
            ctx,
            device_id,
            peers: HashSet::new(),
            streams: MultiStream::new(
                utilities.logger.clone(),
                CudaStreamBackend::new(
                    mem_props,
                    mem_config,
                    mem_alignment,
                    utilities.logger.clone(),
                    stream_priority,
                    captures,
                ),
                max_streams,
            ),
            utilities: Arc::new(utilities),
            collectives: Collectives::new(device_id),
            graphs: Captures::default(),
        }
    }

    fn command_no_inputs(&mut self, stream_id: StreamId) -> Result<Command<'_>, ServerError> {
        self.command(stream_id, [].into_iter())
    }

    fn unsafe_set_current(&self) {
        // TODO: Should check if on the same thread before calling it, since now we don't switch
        // thread except for device memory transfer.
        self.ctx.unsafe_set_current().unwrap();
    }

    fn command<'a>(
        &mut self,
        stream_id: StreamId,
        handles: impl Iterator<Item = &'a BufferBinding>,
    ) -> Result<Command<'_>, ServerError> {
        self.unsafe_set_current();
        let streams = self.streams.resolve(stream_id, handles)?;
        Ok(Command::new(&mut self.ctx, streams, self.utilities.service))
    }

    /// Compile `kernel` if this is the first launch of it, and return it
    /// loaded — or `None` when that failed, in which case the outputs the
    /// launch was given now carry the compilation error.
    ///
    /// Compilation comes first — memoized, so a launch after the first pays a
    /// map lookup — because the write scope stages what the compiled kernel
    /// says it writes. A kernel that fails to compile has no IR and no
    /// compiled answer, so the caller's declared IO decides: only the
    /// declared outputs are left carrying the failure, never the buffers the
    /// kernel was only going to read — tainting those would refuse every
    /// later launch that shares them, an autotune sweep above all.
    ///
    /// A discarded launch claims none. It was never going to write, so a failure in it
    /// leaves nothing stale, and tainting its buffers would fail unrelated
    /// reads of memory the run deliberately left alone.
    fn load_or_fail(
        &mut self,
        kernel: Box<dyn CubeKernel>,
        id: &ArtifactId<()>,
        bindings: &KernelArguments,
        stream_id: StreamId,
        launch_mode: LaunchMode,
    ) -> Option<CudaCompiledKernel> {
        let err = match self.ctx.load_kernel(&*kernel, id, &self.streams.logger) {
            Ok(loaded) => return Some(loaded),
            Err(err) => err,
        };
        if !launch_mode.discards_launch() {
            // No compiled answer exists for a kernel that never compiled, so
            // the caller's declared IO decides what the failure claims: only
            // the outputs, never the buffers the kernel was only going to
            // read — tainting those would refuse every later launch that
            // shares them, an autotune sweep above all.
            let mut written = self.write_set();
            written.extend(bindings.buffers_written(None).cloned());
            let err = ServerError::Launch(err);
            PeerLaunch::new(bindings, None).failed(&err);
            failed_writing(self, stream_id, written, err);
        } else {
            self.profile_failure(&ServerError::Launch(err));
        }
        None
    }

    /// The stream a profiling window records its events into.
    ///
    /// `entry_point` is the server call asking, and it opens the refusal the
    /// way every other graph-state error names the call it turned down. Both
    /// ends of a window come through here: a capture can begin inside an open
    /// window, so the refusal `end_profile` gets is the one `start_profile`
    /// did not.
    ///
    /// # Errors
    ///
    /// A stream recording a graph. Its events would be recorded into the graph
    /// as nodes rather than stamped as they are queued, so the window would
    /// measure nothing and reading it back would abort the capture. Refusing is
    /// what the client's profiled launch path expects here — it hands the
    /// launch back and runs it unmeasured.
    fn profiled_stream(
        &mut self,
        stream_id: StreamId,
        entry_point: &'static str,
    ) -> Result<cudarc::driver::sys::CUstream, ServerError> {
        self.unsafe_set_current();
        let mut streams = self.streams.resolve(stream_id, [].into_iter())?;
        let stream = streams.current();

        if stream.capturing.is_recording() {
            return Err(ServerError::graph_state(format!(
                "{entry_point}: a capture window records launches only, so a profiling event \
                 has no recorded form"
            )));
        }

        Ok(stream.sys)
    }

    /// Mark every open profile invalid: a failure inside a profiling window
    /// invalidates the measurement, and this is what keeps a tuning candidate
    /// that failed from benchmarking at close to zero and winning the tune. A
    /// no-op with no profile open.
    fn profile_failure(&mut self, error: &ServerError) {
        self.ctx.profiler.failure(error);
    }

    /// The grid dimensions this launch runs with, host-read from the count
    /// buffer when the count is dynamic.
    ///
    /// Resolved before the launch's scope opens — see the call site — so the
    /// count buffer is checked here the way the scope checks every other
    /// read: grid dimensions taken from bytes a failure left unwritten
    /// dispatch an absurd grid or scatter into memory that carried no
    /// failure at all.
    ///
    /// # Errors
    ///
    /// The failure the count buffer carries, and the readback's own.
    fn resolve_cube_count(
        &mut self,
        count: CubeCount,
        stream_id: StreamId,
    ) -> Result<(u32, u32, u32), ServerError> {
        match count {
            CubeCount::Static(x, y, z) => Ok((x, y, z)),
            // TODO: CUDA doesn't have an exact equivalent of dynamic dispatch. Instead, kernels are free to launch other kernels.
            // One option is to create a dummy kernel with 1 thread that launches the real kernel with the dynamic dispatch settings.
            // For now, just read the dispatch settings from the buffer.
            CubeCount::Dynamic(binding) => {
                self.streams.ensure_written([&binding].into_iter())?;
                let mut command = self.command(stream_id, [&binding].into_iter())?;
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
                Ok((data[0], data[1], data[2]))
            }
        }
    }

    fn launch_checked(
        &mut self,
        kernel_id: KernelId,
        kernel: &CudaCompiledKernel,
        count: (u32, u32, u32),
        bindings: KernelArguments,
        stream_id: StreamId,
    ) -> Result<(), ServerError> {
        let address_type = kernel_id.address_type;
        let grid_constants = self
            .ctx
            .compilation_options()
            .cpp
            .supports_features
            .grid_constants;
        for peer in bindings.peers() {
            self.enable_peer(peer.binding().service.device)?;
        }
        // The kernel argument a peer buffer is passed through: the driver
        // copies the arguments when the launch is enqueued, so these only have
        // to outlive the call.
        let peer_addresses: Vec<u64> = bindings.peers().map(PeerBinding::address).collect();
        let mut peer_slots = peer_addresses.iter();
        let mut peers = Vec::with_capacity(peer_addresses.len());

        let mut command = self.command(stream_id, bindings.buffers())?;

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

        let mut resources = SmallVec::<[_; 5]>::with_capacity(bindings.resources.len());
        // Tensor maps are owned by the launch function, so they need to be kept around until CUDA
        // has read them. They're individually boxed to keep pointers stable, but we may need to
        // refine this to minimize allocations.
        let mut tensor_maps = Vec::new();

        // Resolving is also where a discarded launch's deferred allocations get their
        // device backing, so this can fail on a device the measured plan does
        // not fit — reported, not panicked.
        for resource in bindings.resources.into_iter() {
            match resource {
                KernelResource::Buffer(binding) => {
                    let resource = command.resource(binding)?;
                    resources.push(resource.binding);
                }
                KernelResource::TensorMap(TensorMapBinding { map, binding }) => {
                    let resource = command.resource(binding)?;
                    let device_ptr = resource.ptr as *mut c_void;

                    let tensor_map = create_tensor_map(map, device_ptr, address_type)?;
                    resources.push(&*tensor_map as *const _ as *mut c_void);
                    tensor_maps.push(tensor_map);
                }
                KernelResource::Peer(peer) => {
                    let slot = peer_slots.next().expect("one slot per peer binding");
                    resources.push(std::ptr::from_ref(slot) as *mut c_void);
                    peers.push(peer);
                }
            }
        }

        if let Some(binding) = info_binding {
            resources.push(command.resource(binding.binding())?.binding);
        }
        resources.extend(info_const);

        command.kernel(&kernel_id, kernel, count, &mut resources)?;
        command.hold_peers(peers);

        Ok(())
    }

    pub(crate) fn utilities(&self) -> Arc<ServerUtilities> {
        self.utilities.clone()
    }
}

fn create_tensor_map(
    map: TensorMapMeta,
    device_ptr: *mut c_void,
    address_type: AddressType,
) -> Result<Box<CUtensorMap>, ServerError> {
    let mut map_ptr = Box::new_uninit();

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
        .map(|s| *s as u64 * map.elem_ty.expand_size(address_type) as u64)
        .collect();
    let elem_stride: Vec<_> = map.elem_stride.iter().rev().map(|s| *s as u32).collect();

    match &map.format {
        // SAFETY: `map_ptr` is a zeroed `MaybeUninit<CUtensorMap>`. `device_ptr` is a
        // valid device pointer. Shape, strides, tile_size, and elem_stride vectors
        // are constructed from validated metadata and outlive this call.
        TensorMapFormat::Tiled(TiledArgs { tile_size }) => unsafe {
            let tile_size: Vec<_> = tile_size.iter().rev().copied().map(|s| s as u32).collect();

            cuTensorMapEncodeTiled(
                map_ptr.as_mut_ptr(),
                elem_to_tensor_map_type(map.elem_ty),
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
                    check_tma_generic(&map, device_ptr, &shape, &strides, &elem_stride).err();
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
            let lower_corner: Vec<_> = args.pixel_box_lower_corner.iter().rev().copied().collect();
            let upper_corner: Vec<_> = args.pixel_box_upper_corner.iter().rev().copied().collect();

            cuTensorMapEncodeIm2col(
                map_ptr.as_mut_ptr(),
                elem_to_tensor_map_type(map.elem_ty),
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
                    check_tma_generic(&map, device_ptr, &shape, &strides, &elem_stride).err();
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
            use cudarc::driver::sys::{CUtensorMapIm2ColWideMode, cuTensorMapEncodeIm2colWide};
            cuTensorMapEncodeIm2colWide(
                map_ptr.as_mut_ptr(),
                elem_to_tensor_map_type(map.elem_ty),
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
                    check_tma_generic(&map, device_ptr, &shape, &strides, &elem_stride).err();
                generic_err.unwrap_or_else(|| LaunchError::Unknown {
                    reason: format!("{err}"),
                    backtrace: BackTrace::capture(),
                })
            })?;
        },
        #[cfg(not(cuda_12080))]
        TensorMapFormat::Im2colWide(_) => {
            return Err(LaunchError::Unknown {
                reason: "CUDA version 12.8 required for tensor map format Im2colWide".into(),
                backtrace: BackTrace::capture(),
            }
            .into());
        }
    };
    // SAFETY: `map_ptr` was fully initialized by one of the `cuTensorMapEncode*`
    // calls above, which all succeeded (errors are propagated before reaching here).
    Ok(unsafe { map_ptr.assume_init() })
}

fn elem_to_tensor_map_type(ty: ElemType) -> CUtensorMapDataType {
    use cudarc::driver::sys::CUtensorMapDataType::*;
    match ty {
        // packed fp4 should be treated as single 4-bit values to simplify indexing/shape handling
        // So a tile of width 16 with fp4 elements is 8 x fp4x2 elements wide.
        #[cfg(cuda_12080)]
        ElemType::Float(FloatKind::E2M1x2) => CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
        ElemType::Float(kind) => match kind {
            // There's no special handling for FP8, so load as u8. `0u8 == 0.0` when reinterpreting.
            FloatKind::E2M1 // single fp4s are padded to a full byte
            | FloatKind::E2M1x2
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
        ElemType::Int(kind) => match kind {
            // UInt is fine because zero bits and size is the same between both
            IntKind::I8 => CU_TENSOR_MAP_DATA_TYPE_UINT8,
            IntKind::I16 => CU_TENSOR_MAP_DATA_TYPE_UINT16,
            IntKind::I32 => CU_TENSOR_MAP_DATA_TYPE_INT32,
            IntKind::I64 => CU_TENSOR_MAP_DATA_TYPE_INT64,
        },
        ElemType::UInt(kind) => match kind {
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
    if matches!(map.elem_ty, ElemType::Float(FloatKind::E2M1x2)) {
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
            map.elem_ty.is_float(),
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
    let tile_size_0_bytes = tile_size[0] as usize * map.elem_ty.size();
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

impl ServerStorage for CudaServer {
    type Storage = GpuStorage;

    fn get_resource(
        &mut self,
        binding: BufferBinding,
        stream_id: StreamId,
    ) -> Result<ManagedResource<GpuResource>, ServerError> {
        // The same claim check a read makes: a buffer a failed launch never
        // filled reports the failure rather than handing back a pointer to
        // whatever was there before.
        self.streams.ensure_written([&binding].into_iter())?;
        let mut command = self.command(stream_id, [&binding].into_iter())?;
        Ok(command.managed_resource(binding)?)
    }
}
