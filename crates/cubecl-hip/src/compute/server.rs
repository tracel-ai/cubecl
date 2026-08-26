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
    compiler::HipCompiler,
    compute::{Command, context::HipContext, stream::HipStreamBackend},
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
    stream::{ExecuteScope, FailureStore, MultiStream, StreamCapture, WriteScoped, failed_writing},
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
        // The bytes are only as good as the work that wrote them.
        if let Err(err) = self
            .streams
            .ensure_written(descriptors.iter().map(|d| &d.handle))
        {
            return Box::pin(async move { Err(err) });
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
            ExecuteScope::over(self, stream_id, written).execute(|server| {
                let mut command = server.command(stream_id, [&descriptor.handle].into_iter());
                command.write_to_gpu(descriptor, data).map_err(Into::into)
            });
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
        if self.compile_failed(&kernel_id, kernel, &bindings, stream_id, launch_mode) {
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
                written.extend(bindings.buffers_written(io.as_deref()).cloned());
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
        written.extend(bindings.buffers_written(io.as_deref()).cloned());
        ExecuteScope::launching(
            self,
            kernel_id.clone(),
            stream_id,
            bindings.buffers_read(io.as_deref()),
            written,
        )
        .execute(|server| server.launch_checked(kernel_id, count, bindings, stream_id));
    }

    fn check(
        &mut self,
        handles: Vec<BufferBinding>,
        _stream_id: StreamId,
    ) -> Result<(), ServerError> {
        self.streams.ensure_written(handles.iter())
    }

    fn flush(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        // A flush reports nothing: a failure lives on the buffers the work
        // left unwritten, and a read of one of them is what surfaces it.
        let mut command = self.command_no_inputs(stream_id);
        command.flush_drops();
        command.stream().memory_management_gpu.storage().flush();

        Ok(())
    }

    fn graph_prepare(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        let mut command = self.command_no_inputs(stream_id);
        Window::on(command.stream()).prepare(stream_id)
    }

    fn begin_capture(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        let mut command = self.command_no_inputs(stream_id);
        Window::on(command.stream()).begin()
    }

    fn end_capture(&mut self, stream_id: StreamId) -> Result<GraphId, ServerError> {
        let id = GraphId::new();
        let instantiated = {
            let mut command = self.command_no_inputs(stream_id);
            Window::on(command.stream()).instantiate(stream_id, id)
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
                let mut streams = server.streams.resolve(stream_id, [].into_iter());
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
        let mut streams = self.streams.resolve(stream_id, [].into_iter());
        self.graphs.destroy(graph, streams.current());
        drop(streams);
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

    fn on_failure(&mut self, _stream: StreamId, error: &ServerError) {
        self.profile_failure(error);
    }

    fn capturing(&mut self, stream: StreamId) -> Option<&mut StreamCapture> {
        self.streams
            .try_stream_mut(&stream)
            .map(|stream| &mut stream.capturing)
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
    /// that failed — in which case the outputs the launch was given now
    /// carry the compilation error.
    ///
    /// Compilation comes first — memoized, so a launch after the first pays a
    /// map lookup — because the write scope stages what the compiled kernel
    /// says it writes. A kernel that fails to compile has no IR and no
    /// compiled answer, so the caller's declared IO decides: only the
    /// declared outputs are left carrying the failure, never the buffers the
    /// kernel was only going to read — tainting those would refuse every
    /// later launch that shares them, an autotune sweep above all.
    ///
    /// A dry run claims none. It was never going to write, so a failure in it
    /// leaves nothing stale, and tainting its buffers would fail unrelated
    /// reads of memory the run deliberately left alone.
    fn compile_failed(
        &mut self,
        kernel_id: &KernelId,
        kernel: <Self as ComputeServer>::Kernel,
        bindings: &KernelArguments,
        stream_id: StreamId,
        launch_mode: LaunchMode,
    ) -> bool {
        if self.ctx.is_loaded(kernel_id) {
            return false;
        }
        let logger = self.streams.logger.clone();
        let Err(err) = self.ctx.compile_kernel(kernel_id, kernel, logger) else {
            return false;
        };
        if !launch_mode.is_skipped() {
            // No compiled answer exists for a kernel that never compiled, so
            // the caller's declared IO decides what the failure claims: only
            // the outputs, never the buffers the kernel was only going to
            // read — tainting those would refuse every later launch that
            // shares them, an autotune sweep above all.
            let mut written = self.write_set();
            written.extend(bindings.buffers_written(None).cloned());
            failed_writing(self, stream_id, written, ServerError::Launch(err));
        } else {
            self.profile_failure(&ServerError::Launch(err));
        }
        true
    }

    /// Mark every open profile invalid: a failure inside a profiling window
    /// invalidates the measurement, and this is what keeps a tuning candidate
    /// that failed from benchmarking at close to zero and winning the tune. A
    /// no-op with no profile open.
    fn profile_failure(&mut self, error: &ServerError) {
        self.ctx.timestamps.failure(error);
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
            // TODO: HIP doesn't have an exact equivalent of dynamic dispatch. Instead, kernels are free to launch other kernels.
            // One option is to create a dummy kernel with 1 thread that launches the real kernel with the dynamic dispatch settings.
            // For now, just read the dispatch settings from the buffer.
            CubeCount::Dynamic(binding) => {
                self.streams.ensure_written([&binding].into_iter())?;
                let mut command = self.command(stream_id, [&binding].into_iter());
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
        count: (u32, u32, u32),
        bindings: KernelArguments,
        stream_id: StreamId,
    ) -> Result<(), ServerError> {
        let mut command = self.command(stream_id, bindings.buffers());

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
    let stream = command.stream();
    let mode = stream.capturing.cache_mode();
    match stream.info_cache.lookup(mode, &words) {
        Lookup::Hit(handle) => Ok(handle),
        Lookup::Build { store } => {
            let handle = command.create_with_data(bytemuck::cast_slice(&words))?;
            if store {
                command.stream().info_cache.store(words, handle.clone());
            }
            Ok(handle)
        }
    }
}
