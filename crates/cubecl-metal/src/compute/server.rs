//! The Metal compute server: the device's streams, its compiled pipelines,
//! and the graphs it has captured.
//!
//! Every entry point here is a [`Server`] method, and the ones that write
//! device memory run inside a [write scope](WriteScoped) — the taint
//! bookkeeping is not spelled out on the failure paths, because a path that
//! forgot it would be silent.

use crate::compute::driver::MetalBindings;
use crate::compute::{Captures, Command, Window};
use crate::{
    compute::context::MetalContext, compute::pipelines::MetalCompiledKernel,
    compute::stream::MetalStreamBackend, memory::MetalStorage,
};
use cubecl_common::{
    bytes::Bytes,
    profile::{Duration, Instant, ProfileDuration, ProfileTicks},
};
use cubecl_core::server::{DeviceCaptures, ServerStorage};
use cubecl_core::{
    MemoryConfiguration,
    prelude::*,
    server::{
        BufferBinding, CopyDescriptor, IoError, KernelArguments, KernelResource, ProfileError,
        ProfilingToken, ServerCommunication, ServerError, ServerUtilities,
    },
};
use cubecl_environment::future::{self, DynFut};
use cubecl_environment::stream::StreamId;
use cubecl_server::command::{DeviceStream, Refused};
use cubecl_server::compiler::ArtifactId;
use cubecl_server::{
    execution::LaunchMode,
    id::GraphId,
    kernel::CubeKernel,
    logging::ServerLogger,
    memory_management::{ManagedMemoryHandle, MemoryAllocationMode, StreamMemoryReport},
    server::Server,
    storage::{ComputeStorage, ManagedResource},
    stream::{ExecuteScope, FailureStore, MultiStream, StreamCapture, WriteScoped, failed_writing},
    timestamp_profiler::TimestampProfiler,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLDevice};
use std::sync::Arc;

/// Metal compute server.
#[derive(Debug)]
pub struct MetalServer {
    context: MetalContext,
    streams: MultiStream<MetalStreamBackend>,
    pub(crate) utilities: Arc<ServerUtilities>,
    timestamps: TimestampProfiler,
    /// The graphs this server has captured — see [`Captures`].
    graphs: Captures,
}

impl MetalServer {
    pub fn new(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        mem_props: cubecl_ir::MemoryDeviceProperties,
        mem_config: MemoryConfiguration,
        utilities: Arc<ServerUtilities>,
        captures: DeviceCaptures,
    ) -> Self {
        let logger = utilities.logger.clone();

        let mut compilation_options = cubecl_cpp::shared::CompilationOptions::default();
        // Metal honors per-op fast math via the `fast::` namespace (MSL 3+).
        compilation_options.supports_features.fast_math = true;
        let context = MetalContext::new(
            device.clone(),
            (*utilities.properties).clone(),
            compilation_options,
        );

        let backend =
            MetalStreamBackend::new(device, mem_props, mem_config, logger.clone(), captures);

        let config = {
            use cubecl_server::config::RuntimeConfig;
            cubecl_server::config::CubeClRuntimeConfig::get()
        };
        let max_streams = config.streaming.max_streams;

        Self {
            context,
            streams: MultiStream::new(logger, backend, max_streams),
            utilities,
            timestamps: TimestampProfiler::default(),
            graphs: Captures::default(),
        }
    }
}

// SAFETY: Only accessed from the server thread. GPU work is serialized through command queue ordering.
unsafe impl Send for MetalServer {}

impl MetalServer {
    fn command_no_inputs(&mut self, stream_id: StreamId) -> Result<Command<'_>, ServerError> {
        self.command(stream_id, [].into_iter())
    }

    fn command<'a>(
        &mut self,
        stream_id: StreamId,
        handles: impl Iterator<Item = &'a BufferBinding>,
    ) -> Result<Command<'_>, ServerError> {
        let streams = self.streams.resolve(stream_id, handles)?;
        Ok(Command::new(
            &mut self.context,
            streams,
            self.utilities.service,
        ))
    }

    /// Mark every open profile invalid: a failure inside a profiling window
    /// invalidates the measurement. A no-op with no profile open.
    fn profile_failure(&mut self, error: &ServerError) {
        self.timestamps.failure(error);
    }

    /// Compile `kernel` if this is the first launch of it, and return its
    /// pipeline — or `None` when that failed, in which case the outputs the
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
    /// A discarded launch claims none. It was never going to write, so a
    /// failure in it leaves nothing stale, and tainting its buffers would fail
    /// unrelated reads of memory the run deliberately left alone.
    fn load_or_fail(
        &mut self,
        kernel: Box<dyn CubeKernel>,
        id: &ArtifactId<()>,
        bindings: &KernelArguments,
        stream_id: StreamId,
        launch_mode: LaunchMode,
    ) -> Option<MetalCompiledKernel> {
        let err = match self.context.load(&*kernel, id, &self.utilities.logger) {
            Ok(compiled) => return Some(compiled),
            Err(err) => err,
        };
        if !launch_mode.discards_launch() {
            let mut written = self.write_set();
            written.extend(bindings.buffers_written(None).cloned());
            failed_writing(self, stream_id, written, ServerError::Launch(err));
        } else {
            self.profile_failure(&ServerError::Launch(err));
        }
        None
    }

    /// The grid dimensions this launch runs with, host-read from the count
    /// buffer when the count is dynamic.
    ///
    /// Read on the host rather than dispatched indirectly, because a captured
    /// graph's commands carry their grid by value: one way to launch, inside
    /// a capture window and out of it.
    ///
    /// Resolved before the launch's scope opens — see the call site — so the
    /// count buffer is checked here the way the scope checks every other
    /// read: grid dimensions taken from bytes a failure left unwritten
    /// dispatch an absurd grid or scatter into memory that carried no
    /// failure at all.
    ///
    /// # Errors
    ///
    /// The failure the count buffer carries, and the readback's own — which
    /// includes the refusal inside a capture window, where the count a later
    /// replay needs is not known yet.
    fn resolve_cube_count(
        &mut self,
        count: CubeCount,
        stream_id: StreamId,
    ) -> Result<(u32, u32, u32), ServerError> {
        match count {
            CubeCount::Static(x, y, z) => Ok((x, y, z)),
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
        kernel: &MetalCompiledKernel,
        count: (u32, u32, u32),
        bindings: KernelArguments,
        stream_id: StreamId,
    ) -> Result<(), ServerError> {
        let mut command = self.command(stream_id, bindings.buffers())?;

        let KernelArguments {
            resources, info, ..
        } = bindings;

        let buffers = resources
            .into_iter()
            .map(|resource| match resource {
                KernelResource::Buffer(binding) => command.resource(binding),
                KernelResource::TensorMap(_) => panic!("Tensor maps not supported on Metal"),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut bindings = MetalBindings {
            buffers,
            info: info.data,
        };

        command.kernel(&kernel_id, kernel, count, &mut bindings)?;

        Ok(())
    }
}

impl ServerCommunication for MetalServer {}

impl WriteScoped for MetalServer {
    type Streams = MultiStream<MetalStreamBackend>;

    fn write_streams(&mut self) -> &mut Self::Streams {
        &mut self.streams
    }

    fn on_failure(&mut self, _stream: StreamId, error: &ServerError) {
        // Measured per device on this backend, so the scope's stream does not
        // narrow which measurement a failure invalidates.
        self.profile_failure(error);
    }

    fn capturing(&mut self, stream: StreamId) -> Option<&mut StreamCapture> {
        self.streams
            .try_stream_mut(&stream)
            .map(|stream| &mut stream.capturing)
    }
}

impl Server for MetalServer {
    fn logger(&self) -> Arc<ServerLogger> {
        self.utilities.logger.clone()
    }

    fn utilities(&self) -> Arc<ServerUtilities> {
        self.utilities.clone()
    }

    fn staging(
        &mut self,
        _sizes: &[usize],
        _stream_id: StreamId,
    ) -> Result<Vec<Bytes>, ServerError> {
        // Unnecessary: shared storage gives direct CPU access to GPU buffers.
        Err(IoError::UnsupportedIoOperation {
            backtrace: cubecl_environment::backtrace::BackTrace::capture(),
        }
        .into())
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

    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        // Buffers another stream wrote are only as good as the work that wrote
        // them; see `FailureStore::ensure_written`.
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

    fn write(&mut self, descriptors: Vec<(CopyDescriptor, Bytes)>, stream_id: StreamId) {
        // Each copy runs in its own scope over its destination: the copy
        // fills it, which is what releases an earlier failure's hold on
        // it — a caller recovers by writing from the host as much as by
        // relaunching — and a failure leaves it as it was, which is what
        // a later read of it has to fail on.
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
            self.context.queue(kernel);
            return;
        }
        let id = ArtifactId {
            kernel: kernel.id(),
            variant: (),
        };
        let Some(compiled) = self.load_or_fail(kernel, &id, &bindings, stream_id, launch_mode)
        else {
            return;
        };
        // A discarded launch stops right here, after compilation and before
        // anything that touches a buffer.
        if launch_mode.discards_launch() {
            return;
        }
        let kernel_id = id.kernel;
        let io = compiled.io.clone();

        // The count resolves before the scope opens, because entering the
        // scope replaces whatever claim the outputs carry — and a count that
        // resolves to zero dispatches nothing, so those claims must be left
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
        // Zero threadgroups writes nothing — no scope opens, so a claim an
        // earlier failure holds on the outputs stays exactly where it was.
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
        .execute(|server| server.launch_checked(kernel_id, &compiled, count, bindings, stream_id));
    }

    fn sync(
        &mut self,
        handles: Vec<BufferBinding>,
        stream_id: StreamId,
    ) -> DynFut<Result<(), ServerError>> {
        // The claim check a read would have made, without the read; claims
        // are set at enqueue time, so they are already in place.
        if let Err(err) = self.streams.ensure_written(handles.iter()) {
            return Box::pin(async move { Err(err) });
        }
        match self.command_no_inputs(stream_id) {
            Ok(mut command) => command.sync(),
            Err(err) => Box::pin(async move { Err(err) }),
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
        self.context.compile_queued(&self.utilities.logger);
    }

    fn flush(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        // A flush reports nothing: a failure lives on the buffers the work
        // left unwritten, and a read of one of them is what surfaces it.
        let mut command = self.command_no_inputs(stream_id)?;
        command.stream().queue().flush();
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
        // lands releases the claim.
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
        let synced = future::block_on(self.sync(Vec::new(), stream_id));
        let stream = self.streams.try_stream_mut(&stream_id);
        self.graphs.destroy(graph, stream);
        if let Err(err) = synced {
            failed_writing(self, stream_id, written, err);
        }
    }

    fn start_profile(&mut self, stream_id: StreamId) -> Result<ProfilingToken, ServerError> {
        self.streams
            .resolve(stream_id, [].into_iter())?
            .current()
            .reject_while_recording("start_profile")?;
        // Drain prior work so the window only contains command buffers committed from here on.
        if let Err(err) = future::block_on(self.sync(Vec::new(), stream_id)) {
            log::warn!("{err}");
        }
        let mut streams = self.streams.resolve(stream_id, [].into_iter())?;
        let token = self.timestamps.start();
        streams.current().queue().profile_windows.open(&token);
        Ok(token)
    }

    fn end_profile(
        &mut self,
        stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        // Flush the final batch and wait for GPU completion so timestamps are valid.
        if let Err(err) = future::block_on(self.sync(Vec::new(), stream_id)) {
            self.timestamps.failure(&err);
        }

        // The window closes whatever the token says below: left open, it
        // would retain every later flush's command buffer — and a failed
        // profile is exactly when the token errors, since a failure anywhere
        // in the window lands in it.
        let buffers = match self.streams.resolve(stream_id, [].into_iter()) {
            Ok(mut streams) => streams.current().queue().profile_windows.close(&token),
            Err(err) => {
                self.timestamps.abandon(token);
                return Err(ProfileError::from(&err));
            }
        };

        // Clear the token (propagates any recorded profiling error). Its system-time result
        // is discarded in favor of GPU timestamps below.
        self.timestamps.stop(token)?;

        // `sync()` waits on the stream's shared event, which can be signaled slightly before
        // a command buffer reaches `Completed` status. `GPUStartTime`/`GPUEndTime` are only
        // valid once completed, so wait explicitly before reading them.
        for buffer in &buffers {
            buffer.waitUntilCompleted();
        }

        // GPU wall-time of the window: latest end minus earliest start across the
        // work-bearing command buffers. Empty window => zero, still device-timed.
        let mut start_s = f64::INFINITY;
        let mut end_s = f64::NEG_INFINITY;
        for buffer in &buffers {
            start_s = start_s.min(buffer.GPUStartTime());
            end_s = end_s.max(buffer.GPUEndTime());
        }

        let span = if buffers.is_empty() {
            Duration::ZERO
        } else if !start_s.is_finite() || !end_s.is_finite() {
            log::warn!(
                "Metal device profiling read non-finite GPU timestamps (start={start_s}, end={end_s}); reporting zero"
            );
            Duration::ZERO
        } else {
            Duration::from_secs_f64((end_s - start_s).max(0.0))
        };

        let base = Instant::now();
        let ticks = ProfileTicks::from_start_end(base, base + span);
        Ok(ProfileDuration::new_device_time(async move { ticks }))
    }

    fn abandon_profile(&mut self, stream_id: StreamId, token: ProfilingToken) {
        // Close the window and drop what it retained: left open it would hold
        // every later flush's command buffers. No sync, which is the whole
        // difference from `end_profile`: nothing is going to read these
        // timestamps.
        if let Some(stream) = self.streams.try_stream_mut(&stream_id) {
            stream.queue().profile_windows.close(&token);
        }
        self.timestamps.abandon(token);
    }

    fn memory_report(&mut self, stream_id: StreamId) -> Option<StreamMemoryReport> {
        self.command_no_inputs(stream_id)
            .ok()
            .map(|mut command| command.memory_report())
    }

    fn memory_cleanup(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        self.command_no_inputs(stream_id)?.memory_cleanup()
    }

    fn allocation_mode(&mut self, mode: MemoryAllocationMode, stream_id: StreamId) {
        if let Ok(mut command) = self.command_no_inputs(stream_id) {
            command.stream().device_memory().mode(mode);
        }
    }
}

impl ServerStorage for MetalServer {
    type Storage = MetalStorage;

    fn get_resource(
        &mut self,
        binding: BufferBinding,
        stream_id: StreamId,
    ) -> Result<ManagedResource<<MetalStorage as ComputeStorage>::Resource>, ServerError> {
        // The same claim check a read makes: a buffer a failed launch never
        // filled reports the failure rather than handing back a pointer to
        // whatever was there before.
        self.streams.ensure_written([&binding].into_iter())?;
        let mut command = self.command(stream_id, [&binding].into_iter())?;
        Ok(command.managed_resource(binding)?)
    }
}
