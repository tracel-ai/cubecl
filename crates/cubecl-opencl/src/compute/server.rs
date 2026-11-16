use crate::compute::artifact::OpenClKernel;
use crate::compute::cl::{self, ClError, Event};
use crate::compute::driver::{OpenCl, OpenClContext, OpenClLaunch};
use crate::compute::programs::OpenClPrograms;
use crate::compute::stream::OpenClStreamBackend;
use crate::memory::OpenClStorage;
use cubecl_common::bytes::Bytes;
use cubecl_common::profile::{ProfileDuration, ProfileTicks};
use cubecl_core::server::ServerStorage;
use cubecl_core::{
    prelude::*,
    server::{
        BufferBinding, CopyDescriptor, IoError, KernelArguments, KernelResource, ProfileError,
        ProfilingToken, ServerCommunication, ServerError, ServerUtilities,
    },
};
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::future::{self, DynFut};
use cubecl_environment::stream::StreamId;
use cubecl_server::command::Command;
use cubecl_server::compiler::{ArtifactId, KernelLoader};
use cubecl_server::{
    execution::LaunchMode,
    id::KernelId,
    kernel::{BufferIOAttr, CubeKernel},
    logging::ServerLogger,
    memory_management::{ManagedMemoryHandle, MemoryAllocationMode, StreamMemoryReport},
    server::Server,
    storage::{ComputeStorage, ManagedResource},
    stream::{ExecuteScope, FailureStore, MultiStream, WriteScoped, failed_writing},
    timestamp_profiler::TimestampProfiler,
};
use opencl_sys::{CL_PROFILING_COMMAND_END, CL_PROFILING_COMMAND_START};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// `OpenCL` compute server.
#[derive(Debug)]
pub struct OpenClServer {
    /// Compiles `CubeCL` IR to `OpenCL` C and on to kernels built for the device, caching both.
    kernels: KernelLoader<OpenClPrograms>,
    streams: MultiStream<OpenClStreamBackend>,
    ctx: OpenClContext,
    pub(crate) utilities: Arc<ServerUtilities>,
    timestamps: TimestampProfiler,
    /// Open profiling windows, by token.
    windows: HashMap<ProfilingToken, Window>,
}

// SAFETY: the context and the handles the fields hold are thread-safe OpenCL objects; only the server thread
// uses the server.
unsafe impl Send for OpenClServer {}

/// An open profiling window: the launches on its stream from its opening, timed by their events.
#[derive(Debug)]
struct Window {
    stream: StreamId,
    first: Option<Event>,
    last: Option<Event>,
    failure: Option<ServerError>,
}

impl OpenClServer {
    /// A server on `cl_context`, whose reference it takes over.
    pub(crate) fn new(
        cl_context: opencl_sys::cl_context,
        kernels: KernelLoader<OpenClPrograms>,
        backend: OpenClStreamBackend,
        utilities: Arc<ServerUtilities>,
    ) -> Self {
        let logger = utilities.logger.clone();
        let max_streams = {
            use cubecl_server::config::RuntimeConfig;
            cubecl_server::config::CubeClRuntimeConfig::get()
                .streaming
                .max_streams
        };
        Self {
            kernels,
            streams: MultiStream::new(logger, backend, max_streams),
            ctx: OpenClContext {
                context: cl_context,
                max_units: utilities.properties.hardware.max_units_per_cube,
            },
            utilities,
            timestamps: TimestampProfiler::default(),
            windows: HashMap::new(),
        }
    }

    fn command<'a>(
        &mut self,
        stream_id: StreamId,
        handles: impl Iterator<Item = &'a BufferBinding>,
    ) -> Result<Command<'_, OpenCl>, ServerError> {
        let streams = self.streams.resolve(stream_id, handles)?;
        Ok(Command::new(&mut self.ctx, streams, self.utilities.service))
    }

    /// Fails the windows open on `stream`: what they measure includes the failed work. (The shared
    /// profiler's `failure` fails every open window, a failure on another stream too.)
    fn profile_failure(&mut self, stream: StreamId, error: &ServerError) {
        for window in self
            .windows
            .values_mut()
            .filter(|window| window.stream == stream)
        {
            window.failure.get_or_insert_with(|| error.clone());
        }
    }

    /// Fails what `bindings` writes (every buffer, when the kernel's `io` is unknown).
    fn fail_outputs(
        &mut self,
        stream_id: StreamId,
        bindings: &KernelArguments,
        io: Option<&[BufferIOAttr]>,
        err: ServerError,
    ) {
        let mut written = self.write_set();
        written.extend(bindings.buffers_written(io).cloned());
        failed_writing(self, stream_id, written, err);
    }

    /// The grid of a launch, a dynamic one read back from the device (`OpenCL` has no indirect
    /// dispatch).
    fn groups(
        &mut self,
        count: CubeCount,
        stream_id: StreamId,
    ) -> Result<(u32, u32, u32), ServerError> {
        let binding = match count {
            CubeCount::Static(x, y, z) => return Ok((x, y, z)),
            CubeCount::Dynamic(binding) => binding,
        };
        self.streams.ensure_written([&binding].into_iter())?;
        let mut command = self.command(stream_id, [&binding].into_iter())?;
        let read = command.read_async(vec![CopyDescriptor::new(
            binding,
            [3].into(),
            [1].into(),
            4,
        )]);
        let data = future::block_on(read)?;
        let groups: &[u32] = bytemuck::cast_slice(&data[0]);
        Ok((groups[0], groups[1], groups[2]))
    }

    /// Launches `compiled` on `stream_id`'s stream, timed when a window is open there.
    fn launch_checked(
        &mut self,
        kernel_id: KernelId,
        compiled: &OpenClKernel,
        count: (u32, u32, u32),
        bindings: KernelArguments,
        stream_id: StreamId,
    ) -> Result<(), ServerError> {
        let timed = self
            .windows
            .values()
            .any(|window| window.stream == stream_id);
        let mut command = self.command(stream_id, bindings.buffers())?;
        let KernelArguments {
            resources, info, ..
        } = bindings;
        let buffers = resources
            .into_iter()
            .map(|resource| match resource {
                KernelResource::Buffer(binding) => command.resource(binding),
                // Tensor maps are CUDA's; OpenCL shares no memory between devices, so a peer's
                // binding cannot be read here.
                KernelResource::TensorMap(_) | KernelResource::Peer(_) => {
                    Err(IoError::UnsupportedIoOperation {
                        backtrace: BackTrace::capture(),
                    })
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut launch = OpenClLaunch {
            buffers,
            info,
            timed,
            event: None,
        };
        command.kernel(&kernel_id, compiled, count, &mut launch)?;
        drop(command);
        if let Some(event) = launch.event {
            for window in self
                .windows
                .values_mut()
                .filter(|window| window.stream == stream_id)
            {
                window.first.get_or_insert_with(|| event.clone());
                window.last = Some(event.clone());
            }
        }
        Ok(())
    }
}

/// A read or a sync waited for here, on the server thread, rather than by the caller: the device's
/// work completes sooner than while the caller waits and the server thread idles, at the cost of
/// the server taking no other command (another stream's included) until it returns.
fn waited<T>(wait: impl Future<Output = Result<T, ServerError>>) -> Result<T, ServerError> {
    future::block_on(wait)
}

impl Drop for OpenClServer {
    fn drop(&mut self) {
        // A kernel on one stream may use another stream's pages: every queue drains before any
        // stream frees its memory.
        let ids: Vec<_> = self.streams.stream_ids().collect();
        for id in ids {
            if let Some(stream) = self.streams.try_stream_mut(&id)
                && let Err(err) = cl::finish(stream.queue.raw)
            {
                log::warn!("{err}");
            }
        }
        // The server's reference; the queues and programs hold their own.
        cl::api().clReleaseContext(self.ctx.context);
    }
}

impl ServerCommunication for OpenClServer {}

impl WriteScoped for OpenClServer {
    type Streams = MultiStream<OpenClStreamBackend>;

    fn write_streams(&mut self) -> &mut Self::Streams {
        &mut self.streams
    }

    fn on_failure(&mut self, stream: StreamId, error: &ServerError) {
        self.profile_failure(stream, error);
    }
}

impl Server for OpenClServer {
    fn logger(&self) -> Arc<ServerLogger> {
        self.utilities.logger.clone()
    }

    fn utilities(&self) -> Arc<ServerUtilities> {
        self.utilities.clone()
    }

    fn initialize_memory(
        &mut self,
        memory: ManagedMemoryHandle,
        size: u64,
        stream_id: StreamId,
    ) -> Result<(), ServerError> {
        let result = self
            .command(stream_id, [].into_iter())
            .and_then(|mut command| Ok(command.initialize_memory(memory.clone(), size)?));
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
        let handles = || descriptors.iter().map(|d| &d.handle);
        let read = self
            .streams
            .ensure_written(handles())
            .and_then(|()| self.command(stream_id, handles()))
            .and_then(|mut command| {
                command.stream().synced();
                waited(command.read_async(descriptors))
            });
        Box::pin(async move { read })
    }

    fn write(&mut self, descriptors: Vec<(CopyDescriptor, Bytes)>, stream_id: StreamId) {
        for (descriptor, data) in descriptors {
            let mut written = self.write_set();
            written.push(descriptor.handle.clone());
            ExecuteScope::over(self, stream_id, written).execute(|server| {
                let mut command = server.command(stream_id, [&descriptor.handle].into_iter())?;
                Ok(command.write_to_gpu(descriptor, data)?)
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
            self.kernels.enqueue(kernel, ());
            return;
        }
        let id = ArtifactId {
            kernel: kernel.id(),
            variant: (),
        };
        let compiled = match self.kernels.load(&*kernel, &id, &self.utilities.logger) {
            Ok(compiled) => compiled,
            Err(err) if launch_mode.discards_launch() => {
                return self.profile_failure(stream_id, &ServerError::Launch(err));
            }
            Err(err) => {
                return self.fail_outputs(stream_id, &bindings, None, ServerError::Launch(err));
            }
        };
        if launch_mode.discards_launch() {
            return;
        }
        let kernel_id = id.kernel;
        let io = compiled.io.clone();

        // The grid first: an empty one runs nothing and writes nothing, so its outputs keep what they held,
        // a failure included.
        let count = match self.groups(count, stream_id) {
            Ok(count) => count,
            Err(err) => return self.fail_outputs(stream_id, &bindings, io.as_deref(), err),
        };
        if count.0 == 0 || count.1 == 0 || count.2 == 0 {
            return;
        }

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
        let synced = self
            .streams
            .ensure_written(handles.iter())
            .and_then(|()| self.command(stream_id, [].into_iter()))
            .and_then(|mut command| {
                command.stream().synced();
                waited(command.sync())
            });
        Box::pin(async move { synced })
    }

    fn check(
        &mut self,
        handles: Vec<BufferBinding>,
        _stream_id: StreamId,
    ) -> Result<(), ServerError> {
        self.streams.ensure_written(handles.iter())
    }

    fn compile_queued(&mut self) {
        self.kernels.compile_queued(&self.utilities.logger);
    }

    fn flush(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        let mut command = self.command(stream_id, [].into_iter())?;
        Ok(cl::flush(command.stream().queue.raw)?)
    }

    fn start_profile(&mut self, stream_id: StreamId) -> Result<ProfilingToken, ServerError> {
        let token = self.timestamps.start();
        self.windows.insert(
            token,
            Window {
                stream: stream_id,
                first: None,
                last: None,
                failure: None,
            },
        );
        Ok(token)
    }

    fn end_profile(
        &mut self,
        _stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        let Some(Window {
            stream: origin,
            first,
            last,
            failure,
        }) = self.windows.remove(&token)
        else {
            return self.timestamps.stop(token);
        };
        // The token is the profiler's to clear either way; its own failures are never recorded (they are the
        // window's, above).
        let cleared = self.timestamps.stop(token);
        if let Some(failure) = failure {
            return Err(ProfileError::from(&failure));
        }
        cleared?;
        // From the window's first launch's start to its last's end, on the stream it opened on whichever stream
        // closes it; a window that launched nothing measured nothing.
        let (Some(first), Some(last)) = (first, last) else {
            return Err(ProfileError::NotMeasured {
                backtrace: BackTrace::capture(),
            });
        };
        let span = (|| {
            let mut resolved = self.streams.resolve(origin, std::iter::empty())?;
            cl::flush(resolved.current().queue.raw)?;
            last.wait().map_err(ClError::into_sync)?;
            let start = first.time_ns(CL_PROFILING_COMMAND_START)?;
            let end = last.time_ns(CL_PROFILING_COMMAND_END)?;
            Ok::<_, ServerError>(Duration::from_nanos(end.saturating_sub(start)))
        })();
        let span = span.map_err(|err| ProfileError::from(&err))?;
        let base = Instant::now();
        let ticks = ProfileTicks::from_start_end(base, base + span);
        Ok(ProfileDuration::new_device_time(async move { ticks }))
    }

    fn abandon_profile(&mut self, _stream_id: StreamId, token: ProfilingToken) {
        self.windows.remove(&token);
        self.timestamps.abandon(token);
    }

    fn memory_report(&mut self, stream_id: StreamId) -> Option<StreamMemoryReport> {
        let mut command = self.command(stream_id, [].into_iter()).ok()?;
        Some(command.memory_report())
    }

    fn stream_ids(&self) -> Vec<StreamId> {
        self.streams.stream_ids().collect()
    }

    fn memory_cleanup(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        self.command(stream_id, [].into_iter())?.memory_cleanup()
    }

    fn allocation_mode(&mut self, mode: MemoryAllocationMode, stream_id: StreamId) {
        if let Ok(mut command) = self.command(stream_id, [].into_iter()) {
            command.allocation_mode(mode);
        }
    }
}

impl ServerStorage for OpenClServer {
    type Storage = OpenClStorage;

    fn get_resource(
        &mut self,
        binding: BufferBinding,
        stream_id: StreamId,
    ) -> Result<ManagedResource<<OpenClStorage as ComputeStorage>::Resource>, ServerError> {
        self.streams.ensure_written([&binding].into_iter())?;
        let mut command = self.command(stream_id, [&binding].into_iter())?;
        Ok(command.managed_resource(binding)?)
    }
}
