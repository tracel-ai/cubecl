//! The HIP compute server: the device's streams, its compiled kernels, and
//! the graphs it has captured.
//!
//! Every entry point here is a [`ComputeServer`] method, and the ones that
//! write device memory run inside a [write scope](WriteScoped) — the taint
//! bookkeeping is not spelled out on the failure paths, because a path that
//! forgot it would be silent.

use super::storage::gpu::{GpuResource, GpuStorage};
use crate::compute::{Captures, Window};
use crate::{
    compute::{Command, context::HipContext, fence::Fence, stream::HipStreamBackend},
    runtime::HipCompiler,
};
use cubecl_common::{bytes::Bytes, profile::ProfileDuration};
use cubecl_core::{
    MemoryConfiguration,
    ir::MemoryDeviceProperties,
    prelude::*,
    server::{
        BufferBinding, CopyDescriptor, Handle, KernelArguments, KernelResource, ProfileError,
        ProfilingToken, ServerCommunication, ServerError, ServerUtilities,
    },
};
use cubecl_environment::future;
use cubecl_environment::future::DynFut;
use cubecl_environment::stream::StreamId;
use cubecl_runtime::command::Refused;
use cubecl_runtime::kernel::BufferIOAttr;
use cubecl_runtime::metadata_cache::Lookup;
use cubecl_runtime::{
    allocator::PitchedMemoryLayoutPolicy,
    compiler::CubeTask,
    config::{CubeClRuntimeConfig, RuntimeConfig},
    dry_run::LaunchMode,
    id::GraphId,
    logging::ServerLogger,
    memory_management::{
        InstallMemoryPoolsError, ManagedMemoryHandle, MemoryAllocationMode, MemoryReport,
        MemoryUsage,
    },
    server::ComputeServer,
    storage::{ComputeStorage, ManagedResource},
    stream::{FailureStore, MultiStream, WriteScoped},
};
use std::sync::Arc;

#[derive(Debug)]
pub struct HipServer {
    ctx: HipContext,
    streams: MultiStream<HipStreamBackend>,
    utilities: Arc<ServerUtilities<Self>>,
    /// The graphs this server has captured — see [`Captures`].
    graphs: Captures,
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
        let mut command = self.command_no_inputs(stream_id);

        Ok(sizes
            .iter()
            .map(|size| command.reserve_cpu(*size, None))
            .collect())
    }

    fn initialize_memory(&mut self, memory: ManagedMemoryHandle, size: u64, stream_id: StreamId) {
        // Fatal rather than reported: `initialize_memory` has no error channel,
        // and an allocation that never got its storage cannot be handed back
        // as a taint either — nothing has a binding to it yet.
        let mut command = self.command_no_inputs(stream_id);
        let reserved = command
            .reserve(size)
            .unwrap_or_else(|err| panic!("failed to reserve {size} bytes of device memory: {err}"));
        command
            .bind(reserved, memory)
            .unwrap_or_else(|err| panic!("failed to bind {size} bytes of device memory: {err}"));
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
            .ensure_written(descriptors.iter().map(|d| &d.handle))
        {
            return Box::pin(async move { Err(err) });
        }
        // The one failure no buffer can report: the context itself.
        if let Some(fault) = self.streams.take_fault() {
            return Box::pin(async move { Err(fault) });
        }

        let mut command = self.command(stream_id, descriptors.iter().map(|d| &d.handle));
        Box::pin(command.read_async(descriptors))
    }

    fn write(&mut self, descriptors: Vec<(CopyDescriptor, Bytes)>, stream_id: StreamId) {
        // Each copy runs in its own scope over its destination: the copy
        // fills it on success, which is what releases an earlier failure's
        // hold on it — a buffer a launch left stale is recovered by writing
        // it from the host just as much as by relaunching into it — and
        // leaves it as it was on failure, which is what a later read of it
        // has to fail on.
        //
        // Every descriptor is attempted, however the one before it went. A
        // copy that stops early leaves the destinations it never reached
        // holding whatever was there before and carrying no failure to say
        // so, which is the one outcome this whole design exists to prevent —
        // and the failure of one copy says nothing about the next, which may
        // name a different buffer on a different stream.
        for (descriptor, data) in descriptors {
            let mut written = self.write_set();
            written.push(descriptor.handle.clone());
            let result = self.while_writing(written, |server, _| {
                let mut command = server.command(stream_id, [&descriptor.handle].into_iter());
                command.write_to_gpu(descriptor, data).map_err(Into::into)
            });
            if let Err(err) = result {
                self.profile_failure(&err);
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
        let kernel_id = kernel.id();
        if self.compile_failed(&kernel_id, kernel, &bindings, launch_mode) {
            return;
        }
        // A dry run stops right here, after compilation and before anything
        // that touches a buffer: resolving resources, uploading metadata or
        // reading a dynamic cube count would materialize memory the run
        // exists to leave unmapped.
        if launch_mode.is_skipped() {
            return;
        }
        let io = self.ctx.kernel_io(&kernel_id);
        if self.skip_on_failed_input(&kernel_id, &bindings, io.as_deref(), stream_id) {
            return;
        }

        // The scope taints what the launch writes until the body proves the
        // work enqueued, so a failure — or a panic — anywhere in it leaves a
        // read of those buffers failing on the error rather than copying
        // bytes nothing wrote.
        let mut written = self.write_set();
        written.extend(bindings.buffers_written(io.as_deref()).cloned());
        let result = self.while_writing(written, |server, _| {
            server.launch_checked(kernel_id, count, bindings, stream_id, io.as_deref())
        });
        if let Err(err) = result {
            self.profile_failure(&err);
        }
    }

    fn check(
        &mut self,
        handles: Vec<BufferBinding>,
        _stream_id: StreamId,
    ) -> Result<(), ServerError> {
        self.streams.ensure_written(handles.iter())
    }

    fn flush(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        // A launch failure is not the flush's to report — it lives on the
        // buffers the launch left unwritten. The device fault is: the context
        // itself is broken, and no buffer can say so.
        if let Some(fault) = self.streams.take_fault() {
            return Err(fault);
        }
        let mut command = self.command_no_inputs(stream_id);

        let current = command.streams.current();
        current.drop_queue.flush(|| Fence::new(current.sys));
        current.memory_management_gpu.storage().flush();

        Ok(())
    }

    fn graph_prepare(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        let mut command = self.command_no_inputs(stream_id);
        Window::on(command.streams.current()).prepare(stream_id)
    }

    fn begin_capture(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        let mut command = self.command_no_inputs(stream_id);
        Window::on(command.streams.current()).begin()
    }

    fn end_capture(&mut self, stream_id: StreamId) -> Result<GraphId, ServerError> {
        let id = GraphId::new();
        let instantiated = {
            let mut command = self.command_no_inputs(stream_id);
            Window::on(command.streams.current()).instantiate(stream_id, id)
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
        written.extend(self.graphs.written(graph));
        let result = self.while_writing(written, |server, _| {
            let mut streams = server.streams.resolve(stream_id, [].into_iter());
            server.graphs.replay(graph, streams.current())
        });
        if let Err(err) = &result {
            self.profile_failure(err);
        }
        result
    }

    fn graph_destroy(&mut self, graph: GraphId, stream_id: StreamId) {
        // No-op for an unknown id (e.g. a double release), and nothing to sync
        // for either.
        if !self.graphs.contains(graph) {
            return;
        }
        // Wait for in-flight replays before dropping the executable: `replay`
        // returns at enqueue time, so one may still be running against it. A
        // failed sync means the stream already faulted — so no replay is still
        // running and destroying is safe — but don't silently swallow the
        // error: surface it on the stream so the next op reports it.
        let synced = cubecl_environment::future::block_on(self.sync(Vec::new(), stream_id));
        let mut streams = self.streams.resolve(stream_id, [].into_iter());
        self.graphs.destroy(graph, streams.current());
        drop(streams);
        if let Err(err) = synced {
            // The synchronize itself failed, leaving a context every logical
            // stream sharing it keeps hitting: a device fault, not any
            // buffer's failure.
            self.streams.fault(err);
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
        if let Some(fault) = self.streams.take_fault() {
            return Box::pin(async move { Err(fault) });
        }
        self.command_no_inputs(stream_id).sync()
    }

    fn start_profile(&mut self, stream_id: StreamId) -> Result<ProfilingToken, ServerError> {
        cubecl_environment::future::block_on(self.sync(Vec::new(), stream_id))?;
        Ok(self.ctx.timestamps.start())
    }

    fn end_profile(
        &mut self,
        stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        if let Err(err) = cubecl_environment::future::block_on(self.sync(Vec::new(), stream_id)) {
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
        // The same claim check a read makes: a buffer a failed launch never
        // filled reports the failure rather than handing back a pointer to
        // whatever was there before.
        self.streams.ensure_written([&binding].into_iter())?;
        let mut command = self.command(stream_id, [&binding].into_iter());
        let memory = binding.memory.clone();
        let resource = command.resource(binding)?;

        Ok(ManagedResource::new(memory, resource))
    }

    fn memory_usage(&mut self, stream_id: StreamId) -> MemoryUsage {
        self.command_no_inputs(stream_id).memory_usage()
    }

    fn memory_report(&mut self, stream_id: StreamId) -> MemoryReport {
        self.command_no_inputs(stream_id).memory_report()
    }

    fn stream_ids(&self) -> Vec<StreamId> {
        self.streams.stream_ids().collect()
    }

    fn memory_cleanup(&mut self, stream_id: StreamId) {
        self.command_no_inputs(stream_id).memory_cleanup()
    }

    fn allocation_mode(&mut self, mode: MemoryAllocationMode, stream_id: StreamId) {
        let mut command = self.command_no_inputs(stream_id);
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
        self.command_no_inputs(stream_id)
            .install_memory_pools(config, &props)
    }
}

impl ServerCommunication for HipServer {
    const SERVER_COMM_ENABLED: bool = false;
}

impl WriteScoped for HipServer {
    type Streams = MultiStream<HipStreamBackend>;

    fn write_streams(&mut self) -> &mut Self::Streams {
        &mut self.streams
    }
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
            graphs: Captures::default(),
        }
    }

    fn command_no_inputs(&mut self, stream_id: StreamId) -> Command<'_> {
        self.command(stream_id, [].into_iter())
    }

    fn command<'a>(
        &mut self,
        stream_id: StreamId,
        handles: impl Iterator<Item = &'a BufferBinding>,
    ) -> Command<'_> {
        let streams = self.streams.resolve(stream_id, handles);
        Command::new(&mut self.ctx, streams)
    }

    /// Compile `kernel` if this is the first launch of it, and say whether
    /// that failed — in which case everything the launch was given now
    /// carries the compilation error.
    ///
    /// Compilation comes first — memoized, so a launch after the first pays a
    /// map lookup — because the write scope stages what the compiled kernel
    /// says it writes. A kernel that fails to compile has no IR and no
    /// answer, so every buffer the launch was given is left as it was and all
    /// of them carry the failure.
    ///
    /// A dry run claims none. It was never going to write, so a failure in it
    /// leaves nothing stale, and tainting its buffers would fail unrelated
    /// reads of memory the run deliberately left alone.
    fn compile_failed(
        &mut self,
        kernel_id: &KernelId,
        kernel: <Self as ComputeServer>::Kernel,
        bindings: &KernelArguments,
        launch_mode: LaunchMode,
    ) -> bool {
        if self.ctx.is_loaded(kernel_id) {
            return false;
        }
        let logger = self.streams.logger.clone();
        let Err(err) = self.ctx.compile_kernel(kernel_id, kernel, logger) else {
            return false;
        };
        let error = ServerError::Launch(err);
        self.profile_failure(&error);
        if !launch_mode.is_skipped() {
            let mut written = self.write_set();
            written.extend(bindings.buffers().cloned());
            self.failed_writing(written, error);
        }
        true
    }

    /// Whether a launch reading `bindings` must be skipped because one of its
    /// inputs carries a failure, having claimed everything it would have
    /// written for that same failure.
    ///
    /// Skip, do not taint: a launch whose input cannot be trusted does not
    /// run. Running it is not merely wasted device time — a buffer holding
    /// garbage can be read as a dynamic cube count or as gather indices,
    /// scattering into memory that carried no failure at all. The outputs
    /// take the failure that stopped the launch, exactly as a failed launch's
    /// would, so a read downstream fails on the root cause.
    ///
    /// Except while this stream records a graph: skipping would seal a
    /// recording missing an operation, and the replay contract has the caller
    /// write fresh inputs before each replay — clearing the very taint that
    /// would explain the hole. A tainted input dooms the capture instead, and
    /// `end_capture` refuses to seal it.
    fn skip_on_failed_input(
        &mut self,
        kernel_id: &KernelId,
        bindings: &KernelArguments,
        io: Option<&[BufferIOAttr]>,
        stream_id: StreamId,
    ) -> bool {
        let Some(found) = self.streams.read_failure(bindings.buffers_read(io)) else {
            return false;
        };
        self.profile_failure(&found.error);
        if let Some(stream) = self.streams.try_stream_mut(&stream_id)
            && stream.capturing.is_recording()
        {
            stream.capturing.fail(found.error.clone());
        }
        self.streams
            .propagate(&found, kernel_id.clone(), bindings.buffers_written(io));
        true
    }

    /// Mark every open profile invalid: a failure inside a profiling window
    /// invalidates the measurement, and this is what keeps a tuning candidate
    /// that failed from benchmarking at close to zero and winning the tune. A
    /// no-op with no profile open.
    fn profile_failure(&mut self, error: &ServerError) {
        self.ctx.timestamps.failure(error);
    }

    fn launch_checked(
        &mut self,
        kernel_id: KernelId,
        count: CubeCount,
        bindings: KernelArguments,
        stream_id: StreamId,
        io: Option<&[BufferIOAttr]>,
    ) -> Result<(), ServerError> {
        let mut command = self.command(stream_id, bindings.buffers());

        // A launch being recorded into a graph hands its buffers to the graph:
        // a replay that fails runs none of the recorded launches, so it leaves
        // all of them as they were. A no-op outside a capture window, and never
        // reached by a dry run, which records nothing to answer for.
        let stream = command.streams.current();
        stream
            .capturing
            .record(bindings.buffers_written(io).cloned());

        let count = match count {
            CubeCount::Static(x, y, z) => (x, y, z),
            // TODO: HIP doesn't have an exact equivalent of dynamic dispatch. Instead, kernels are free to launch other kernels.
            // One option is to create a dummy kernel with 1 thread that launches the real kernel with the dynamic dispatch settings.
            // For now, just read the dispatch settings from the buffer.
            CubeCount::Dynamic(binding) => {
                // Inside the write scope, so a failed readback claims what the
                // launch would have written rather than aborting the process.
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

        let KernelArguments {
            resources, info, ..
        } = bindings;

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

        command.kernel(kernel_id, count, &mut resources)?;

        Ok(())
    }
}

/// Build — or reuse from the cache — the device buffer holding a launch's info
/// words.
///
/// The info is read-only metadata with no tensor pointers in it, so sharing a
/// buffer across launches — of different kernels, even — is sound, and it is
/// what lets a stable-shape decode allocate and copy nothing inside a capture
/// window. The cache's [`lookup`] makes every decision; the capture lifecycle
/// drives its mode, so during a capture every buffer is kept and none evicted.
///
/// `words` is taken by value so a miss hands them to the cache as the key
/// without cloning, and the buffer's bytes always equal the key's, so a hit is
/// byte-identical to what the miss would have built.
///
/// [`lookup`]: cubecl_runtime::metadata_cache::MetadataInfoCache::lookup
fn info_buffer(command: &mut Command<'_>, words: Vec<u64>) -> Result<Handle, ServerError> {
    let stream = command.streams.current();
    let mode = stream.capturing.cache_mode();
    match stream.info_cache.lookup(mode, &words) {
        Lookup::Hit(handle) => Ok(handle),
        Lookup::Build { store } => {
            let handle = command.create_with_data(bytemuck::cast_slice(&words))?;
            if store {
                command
                    .streams
                    .current()
                    .info_cache
                    .store(words, handle.clone());
            }
            Ok(handle)
        }
    }
}
