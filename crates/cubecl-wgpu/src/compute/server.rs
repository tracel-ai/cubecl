use std::collections::HashMap;
use std::marker::PhantomData;

use super::graph::WgpuGraph;
use super::storage::{WgpuResource, WgpuStorage};
use crate::WgpuCompiler;
use crate::schedule::{BindingsResource, ScheduleTask, ScheduledWgpuBackend};
use alloc::sync::Arc;
use cubecl_common::pool::LeasePool;
use cubecl_common::{
    bytes::Bytes,
    profile::{ProfileDuration, TimingMethod},
};
use cubecl_core::server::{BufferBinding, KernelResource};
use cubecl_core::zspace::Shape;
use cubecl_core::{
    MemoryConfiguration, WgpuCompilationOptions,
    prelude::*,
    server::{
        CopyDescriptor, IoError, KernelArguments, LaunchError, ProfileError, ProfilingToken,
        ServerCommunication, ServerError, ServerUtilities,
    },
    zspace::{Strides, strides},
};
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::future::DynFut;
#[cfg(feature = "spirv")]
use cubecl_environment::persistence::Store;
use cubecl_environment::stream::StreamId;
use cubecl_ir::MemoryDeviceProperties;
use cubecl_runtime::allocator::ContiguousMemoryLayoutPolicy;
#[cfg(feature = "spirv")]
use cubecl_runtime::compiler::{KernelCacheKey, compilation_store, store_compiled};
use cubecl_runtime::memory_management::{
    InstallMemoryPoolsError, ManagedMemoryHandle, MemoryReport, MemoryUsage, SharedMemoryBindings,
};
use cubecl_runtime::{
    compiler::{CompilationCache, CubeTask},
    config::{CubeClRuntimeConfig, RuntimeConfig},
    dry_run::LaunchMode,
    id::GraphId,
    logging::ServerLogger,
    memory_management::MemoryAllocationMode,
    server::ComputeServer,
    storage::ManagedResource,
    stream::WriteScoped,
    stream::scheduler::{
        SchedulerMultiStream, SchedulerMultiStreamOptions, SchedulerStrategy,
        SchedulerStreamBackend,
    },
    validation::{validate_cube_dim, validate_units},
};
use wgpu::ComputePipeline;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamsTransfer {
    Immediate,
    Uniform,
}

/// Compiler kind and info used when compiling a specific kernel. Used to determine parameter passing strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilerInfo {
    Vulkan { params_transfer: ParamsTransfer },
    Metal,
    WGSL,
    None,
}

/// Wgpu compute server.
#[derive(Debug)]
pub struct WgpuServer<C: WgpuCompiler> {
    pub(crate) device: wgpu::Device,
    // A buffer that can be used to store stream id without extra allocations.
    streams_pool: Vec<StreamId>,
    /// The pipelines built so far, in front of the SPIR-V store when there is
    /// one.
    pipelines: CompilationCache<KernelId, (Arc<ComputePipeline>, CompilerInfo)>,
    scheduler: SchedulerMultiStream<ScheduledWgpuBackend>,
    #[cfg(feature = "spirv")]
    pub(crate) spirv_cache: Option<Store<(u64, KernelCacheKey), cubecl_spirv::SpirvCacheEntry>>,
    #[cfg(feature = "spirv")]
    pub(crate) build_id: cubecl_common::hash::StableHash,
    pub compilation_options: WgpuCompilationOptions,
    pub(crate) backend: wgpu::Backend,
    pub(crate) utilities: Arc<ServerUtilities<Self>>,
    /// Reusable buffers for the cross-stream input bindings of each launch.
    shared_bindings_pool: LeasePool<SharedMemoryBindings>,
    /// Captured graphs owned by this server, keyed by the [`GraphId`] handed to
    /// the client. `end_capture` inserts, `replay` looks up, `graph_destroy`
    /// removes (dropping the [`WgpuGraph`] unpins the buffers it retained).
    graphs: HashMap<GraphId, WgpuGraph>,
    _compiler: PhantomData<C>,
}

impl<C: WgpuCompiler> ServerCommunication for WgpuServer<C> {
    const SERVER_COMM_ENABLED: bool = false;
}

impl<C: WgpuCompiler> WriteScoped for WgpuServer<C> {
    type Streams = SchedulerMultiStream<ScheduledWgpuBackend>;

    fn write_streams(&mut self) -> &mut Self::Streams {
        &mut self.scheduler
    }
}

impl<C: WgpuCompiler> WgpuServer<C> {
    /// Create a new server.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        compilation_options: WgpuCompilationOptions,
        device: wgpu::Device,
        queue: wgpu::Queue,
        tasks_max: usize,
        backend: wgpu::Backend,
        timing_method: TimingMethod,
        utilities: ServerUtilities<Self>,
    ) -> Self {
        #[cfg(feature = "spirv")]
        let adapter_info = device.adapter_info();
        let backend_scheduler = ScheduledWgpuBackend::new(
            device.clone(),
            queue.clone(),
            memory_properties,
            memory_config,
            timing_method,
            backend,
            tasks_max,
            utilities.logger.clone(),
            compilation_options.supports_vulkan_compiler,
        );

        let config = CubeClRuntimeConfig::get();
        let max_streams = config.streaming.max_streams;

        #[cfg(feature = "spirv")]
        let spirv_cache = compilation_store(
            "vulkan",
            format!("spirv_{}_{}", adapter_info.vendor, adapter_info.device),
        );

        // WGSL is compiled by the driver on every run, so without the SPIR-V
        // store there is nothing persisted for a switch to invalidate.
        #[cfg(feature = "spirv")]
        let pipelines = CompilationCache::mirroring(&spirv_cache);
        #[cfg(not(feature = "spirv"))]
        let pipelines = CompilationCache::unbound();

        Self {
            compilation_options,
            streams_pool: Vec::new(),
            device,
            pipelines,
            scheduler: SchedulerMultiStream::new(
                utilities.logger.clone(),
                backend_scheduler,
                SchedulerMultiStreamOptions {
                    max_streams,
                    max_tasks: tasks_max,
                    strategy: SchedulerStrategy::Interleave,
                },
            ),
            #[cfg(feature = "spirv")]
            spirv_cache,
            #[cfg(feature = "spirv")]
            build_id: cubecl_runtime::compiler::build_id_hash(),
            backend,
            utilities: Arc::new(utilities),
            shared_bindings_pool: LeasePool::with_capacity(tasks_max * max_streams as usize),
            graphs: HashMap::new(),
            _compiler: PhantomData,
        }
    }

    fn prepare_bindings(
        &mut self,
        bindings: KernelArguments,
        compiler_info: CompilerInfo,
    ) -> Result<BindingsResource, IoError> {
        // Store all the resources we'll be using. This could be eliminated if
        // there was a way to tie the lifetime of the resource to the memory handle.
        let mut resources = Vec::with_capacity(bindings.resources.len());

        for resource in bindings.resources.into_iter() {
            match resource {
                KernelResource::Buffer(b) => {
                    let stream = self.scheduler.stream(&b.stream);
                    let resource = stream.mem_manage.get_resource(b)?;
                    resources.push(resource);
                }
                KernelResource::TensorMap(_) => panic!("Tensor map not supported in wgpu"),
            }
        }

        Ok(BindingsResource {
            resources,
            info: bindings.info,
            compiler_info,
        })
    }

    fn pipeline(
        &mut self,
        kernel: <Self as ComputeServer>::Kernel,
        bindings: &KernelArguments,
    ) -> Result<(Arc<ComputePipeline>, CompilerInfo), LaunchError> {
        let kernel_id = kernel.id();
        let mode = kernel_id.mode;

        if let Some(pipeline) = self.pipelines.get(&kernel_id) {
            return Ok(pipeline.clone());
        }

        let cached = self.load_cached_pipeline(&kernel_id, bindings, mode)?;

        if let Some(Ok(pipeline)) = cached {
            self.pipelines.insert(kernel_id, pipeline.clone());
            return Ok(pipeline);
        }

        validate_cube_dim(&self.utilities.properties, &kernel_id)?;
        validate_units(&self.utilities.properties, &kernel_id)?;

        let definition = kernel.define();

        let mut compiler = C::init(self.backend, &self.compilation_options);
        let mut compiled = compiler.compile_kernel(self, kernel, definition)?;

        if self.scheduler.logger.compilation_source_activated() {
            compiled.debug_info = Some(DebugInformation::new(
                compiler.lang_tag(),
                kernel_id.clone(),
            ));
        }
        self.scheduler.logger.log_compilation(&compiled);

        compiler.validate_ir(&compiled.repr, &self.utilities.properties)?;
        let (compiler_info, auto_repr) = compiler.normalize_repr(compiled.repr);
        let repr = auto_repr.as_ref().map(|r| r.as_ref());

        // /!\ Do not delete the following commented code.
        // This is useful while working on the metal compiler.
        // Also the errors are printed nicely which is not the case when this is the runtime
        // that does it.
        // {
        //     // Write shader in metal file then compile it for error
        //     std::fs::write("shader.metal", &compiled.source).expect("should write to file");
        //     let status = std::process::Command::new("xcrun")
        //         .args(vec![
        //             "-sdk",
        //             "macosx",
        //             "metal",
        //             "-o",
        //             "shader.ir",
        //             "-c",
        //             "shader.metal",
        //             "-w",
        //         ])
        //         .status()
        //         .expect("should launch the command");
        //     if !status.success() {
        //         println!("SOURCE:\n{}", compiled.source);
        //         std::process::exit(status.code().unwrap());
        //     }
        // }

        let module = self.create_module(
            &compiled.entrypoint_name,
            kernel_id.cube_dim.into(),
            repr,
            &compiled.source,
            mode,
        )?;
        let pipeline = self.create_pipeline(&compiled.entrypoint_name, repr, module, bindings);
        self.pipelines
            .insert(kernel_id.clone(), (pipeline.clone(), compiler_info));

        #[cfg(feature = "spirv")]
        if let Some(Err(key)) = cached
            && let Some(crate::AutoRepresentation::SpirV(kernel)) = auto_repr
        {
            let cache = self.spirv_cache.as_mut().unwrap();
            store_compiled(
                cache,
                key,
                cubecl_spirv::SpirvCacheEntry::new(compiled.entrypoint_name, kernel),
            );
        }

        Ok((pipeline, compiler_info))
    }
}

impl<C: WgpuCompiler> ComputeServer for WgpuServer<C> {
    type Kernel = Box<dyn CubeTask<C>>;
    type Storage = WgpuStorage;
    type MemoryLayoutPolicy = ContiguousMemoryLayoutPolicy;
    type Info = wgpu::Backend;

    fn logger(&self) -> Arc<ServerLogger> {
        self.scheduler.logger.clone()
    }

    fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }

    fn staging(
        &mut self,
        _sizes: &[usize],
        _stream_id: StreamId,
    ) -> Result<Vec<Bytes>, ServerError> {
        // TODO: Check if using a staging buffer is useful here.
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }

    fn initialize_memory(&mut self, memory: ManagedMemoryHandle, size: u64, stream_id: StreamId) {
        let (stream, failures) = self.scheduler.stream_and_failures(&stream_id);
        let reserved = stream
            .empty(size, failures)
            .unwrap_or_else(|err| panic!("failed to reserve {size} bytes of device memory: {err}"));
        stream.mem_manage.bind(reserved, memory, failures);
    }

    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        // A read is a host sync: it cannot be recorded, and the recorded work
        // has not executed, so there is nothing meaningful to read anyway.
        if let Err(err) = self
            .scheduler
            .stream(&stream_id)
            .reject_while_recording("read")
        {
            return Box::pin(async move { Err(err) });
        }

        // Buffers another stream wrote are only as good as the work that wrote
        // them; see `StreamPool::ensure_written`. The reader's own errors are
        // surfaced by `read_resources`' flush further down.
        if let Err(err) = self
            .scheduler
            .ensure_written(descriptors.iter().map(|d| &d.handle))
        {
            return Box::pin(async move { Err(err) });
        }

        let mut streams = vec![stream_id];
        let mut resources = Vec::with_capacity(descriptors.len());
        for desc in descriptors {
            if contiguous_strides(&desc.shape) != desc.strides {
                return Box::pin(async {
                    Err(IoError::UnsupportedStrides {
                        backtrace: BackTrace::capture(),
                    }
                    .into())
                });
            }
            if !streams.contains(&desc.handle.stream) {
                streams.push(desc.handle.stream);
            }
            let stream = self.scheduler.stream(&desc.handle.stream);
            let resource = match stream.mem_manage.get_resource(desc.handle) {
                Ok(val) => val,
                Err(err) => return Box::pin(async move { Err(err.into()) }),
            };
            resources.push((resource, desc.shape, desc.elem_size));
        }

        self.scheduler.execute_streams(streams);

        let (stream, failures) = self.scheduler.stream_and_failures(&stream_id);
        stream.read_resources(resources, stream_id, failures)
    }

    fn write(&mut self, descriptors: Vec<(CopyDescriptor, Bytes)>, stream_id: StreamId) {
        // Writes go on the queue, not the encoder — they cannot be recorded
        // into a software graph (v1; CUDA records them as memcpy nodes).
        //
        // Rejected lazily, and queued on the caller. When the caller is the
        // stream recording the capture, that is what fails its `end_capture`
        // rather than handing back a graph missing an operation. When it is a
        // neighbour sharing the pooled stream, the write was never going into
        // anyone's graph and the refusal is the neighbour's own to surface —
        // failing a capture on it would report one stream's window to another.
        {
            let recording = self
                .scheduler
                .stream(&stream_id)
                .reject_while_recording("write");
            if let Err(err) = recording {
                // Nothing is copied, so every destination this call was given is
                // left as it was — taint them, or a read of one on another
                // logical stream finds no failure to fail on and copies stale
                // bytes.
                self.scheduler.taint(
                    err.clone(),
                    descriptors.iter().map(|(desc, _)| &desc.handle),
                );
                self.scheduler
                    .stream(&stream_id)
                    .errors
                    .push(stream_id, err);
                return;
            }
        }
        for (desc, data) in descriptors {
            // Each copy runs in its own scope over its destination: the write
            // that lands fills it, which is what releases an earlier
            // failure's hold on it — a caller recovers by writing from the
            // host as much as by relaunching — and a failure leaves it as it
            // was, which is what a later read of it has to fail on. The scope
            // queues failures on the caller's stream, the one that flushes
            // them, even though the resource is resolved on the stream that
            // owns the handle.
            let result = self.while_writing(
                stream_id,
                (desc, data),
                |(desc, _), written| written.push(desc.handle.clone()),
                |server, (desc, data), _| {
                    if contiguous_strides(&desc.shape) != desc.strides {
                        return Err(ServerError::Io(IoError::UnsupportedStrides {
                            backtrace: BackTrace::capture(),
                        }));
                    }

                    // The write is registered on the caller, so name the
                    // stream that owns the handle as an argument: its queued
                    // work has to land before this write overwrites the same
                    // memory.
                    let owner = desc.handle.stream;
                    let handle = desc.handle.clone();
                    let stream = server.scheduler.stream(&owner);
                    let resource = stream
                        .mem_manage
                        .get_resource(desc.handle)
                        .map_err(ServerError::Io)?;
                    let task = ScheduleTask::Write {
                        data,
                        buffer: resource,
                        handle,
                    };

                    server.scheduler.register(stream_id, task, &[owner]);
                    Ok(())
                },
            );
            if result.is_err() {
                return;
            }
        }
    }

    fn get_resource(
        &mut self,
        binding: BufferBinding,
        stream_id: StreamId,
    ) -> Result<ManagedResource<WgpuResource>, ServerError> {
        let mut streams = vec![stream_id];
        if binding.stream != stream_id {
            streams.push(binding.stream);
        }
        self.scheduler.execute_streams(streams);
        let stream = self.scheduler.stream(&binding.stream);
        let memory = binding.memory.clone();
        let resource = stream.mem_manage.get_resource(binding)?;

        Ok(ManagedResource::new(memory, resource))
    }

    unsafe fn launch(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        args: KernelArguments,
        stream_id: StreamId,
        launch_mode: LaunchMode,
    ) {
        // The scope taints what the launch writes until the body proves the
        // work enqueued, so a failure — or a panic — anywhere in it leaves a
        // read of those buffers failing on the error rather than copying
        // bytes nothing wrote.
        //
        // A dry run stages none. It was never going to write, so a failure in
        // it leaves nothing stale, and tainting its buffers would fail
        // unrelated reads of memory the run deliberately left alone.
        let _ = self.while_writing(
            stream_id,
            args,
            |args, written| {
                if !launch_mode.is_skipped() {
                    written.extend(args.buffers_written().cloned());
                }
            },
            |server, args, written| {
                let (pipeline, compiler_info) = server
                    .pipeline(kernel, &args)
                    .map_err(ServerError::Launch)?;

                if launch_mode.is_skipped() {
                    return Ok(());
                }

                server.streams_pool.clear();
                // Reuse a pooled buffer to avoid allocating on every launch; it returns to the pool
                // automatically when the guard drops.
                let mut shared_inputs = server.shared_bindings_pool.acquire();
                // Pin the memory of every input that lives on another stream (released in `WgpuStream::flush`).
                args.resources.iter().for_each(|resource| match resource {
                    KernelResource::Buffer(b) => {
                        server.streams_pool.push(b.stream);
                        if b.stream != stream_id {
                            shared_inputs.push(b.memory.clone());
                        }
                    }
                    KernelResource::TensorMap(_) => {
                        panic!("Tensor maps not supported in WGPU")
                    }
                });

                let resources = server
                    .prepare_bindings(args, compiler_info)
                    .map_err(ServerError::Io)?;
                // A launch recorded into a graph hands its buffers to the graph, which
                // answers for them if a replay never runs. A no-op outside a window.
                let stream = server.scheduler.stream(&stream_id);
                stream.capturing.record(written.iter().cloned());

                let task = ScheduleTask::Execute {
                    pipeline,
                    count,
                    resources,
                    shared_inputs,
                };

                server
                    .scheduler
                    .register(stream_id, task, &server.streams_pool);
                Ok(())
            },
        );
    }

    fn flush(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        self.scheduler.execute_streams(vec![stream_id]);

        let (stream, failures) = self.scheduler.stream_and_failures(&stream_id);

        stream.flush(stream_id, failures)
    }

    /// Returns the total time of GPU work this sync completes.
    fn sync(&mut self, stream_id: StreamId) -> DynFut<Result<(), ServerError>> {
        if let Err(err) = self
            .scheduler
            .stream(&stream_id)
            .reject_while_recording("sync")
        {
            return Box::pin(async move { Err(err) });
        }
        self.scheduler.execute_streams(vec![stream_id]);
        let (stream, failures) = self.scheduler.stream_and_failures(&stream_id);

        stream.sync(stream_id, failures)
    }

    fn start_profile(&mut self, stream_id: StreamId) -> Result<ProfilingToken, ServerError> {
        // Recorded launches do not execute, so a profile of the window would
        // measure nothing.
        self.scheduler
            .stream(&stream_id)
            .reject_while_recording("start_profile")?;
        self.scheduler.execute_streams(vec![stream_id]);
        let (stream, failures) = self.scheduler.stream_and_failures(&stream_id);
        stream.start_profile(stream_id, failures)
    }

    fn end_profile(
        &mut self,
        stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        self.scheduler.execute_streams(vec![stream_id]);
        let (stream, failures) = self.scheduler.stream_and_failures(&stream_id);

        stream.end_profile(token, stream_id, failures)
    }

    fn memory_usage(&mut self, stream_id: StreamId) -> Result<MemoryUsage, ServerError> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        Ok(stream.mem_manage.memory_usage())
    }

    fn memory_report(&mut self, stream_id: StreamId) -> Result<MemoryReport, ServerError> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        Ok(stream.mem_manage.memory_report())
    }

    fn stream_ids(&self) -> Vec<StreamId> {
        self.scheduler.stream_ids().collect()
    }

    fn memory_cleanup(&mut self, stream_id: StreamId) {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        // The info cache's buffers are live slices in the uniforms pool; an
        // explicit cleanup exists to leave the pools empty, so every entry not
        // pinned by a live graph goes too (entries are recreated on their next
        // miss).
        stream.info_cache.clear_unpinned();
        let (stream, failures) = self.scheduler.stream_and_failures(&stream_id);
        stream.mem_manage.memory_cleanup(true, failures);
    }

    fn allocation_mode(&mut self, mode: MemoryAllocationMode, stream_id: StreamId) {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.mem_manage.mode(mode);
    }

    fn install_memory_pools(
        &mut self,
        config: MemoryConfiguration,
        stream_id: StreamId,
    ) -> Result<(), InstallMemoryPoolsError> {
        // Streams created from now on build their main pool with the new
        // layout; memory is per stream, so already-created streams keep theirs.
        self.scheduler
            .backend_mut()
            .factory()
            .set_gpu_pools(config.clone());
        let (_, props) = self.scheduler.backend_mut().factory().gpu_pools();

        // The calling stream's pools are rebuilt in place, keeping the old
        // layout when something is still live in them.
        self.scheduler.execute_streams(vec![stream_id]);
        let (stream, failures) = self.scheduler.stream_and_failures(&stream_id);
        stream
            .mem_manage
            .install_memory_pools(config, &props, failures)
    }

    fn graph_prepare(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        // Drain queued tasks first so pre-capture work is not attributed to
        // the capture window.
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);

        stream.capturing.prepare(stream_id)?;

        // Route every allocation from here until `end_capture` into the
        // persistent pools and track the touched slices: warmup populates the
        // pools with the capture run's full working set, the recorded run
        // reuses those slices, and everything it touches is pinned to the
        // graph at `end_capture`. The non-`NoCapture` state also isolates this
        // stream in the scheduler (see `requires_isolation`).
        stream.mem_manage.capture_begin();
        Ok(())
    }

    fn begin_capture(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        // Materialize the warmup work queued in the scheduler before the
        // recording window opens.
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);

        stream.capturing.begin()?;

        // Submit the warmup work and surface its queued errors now, so a
        // warmup failure is reported here — where the diagnostic points at the
        // cause — instead of failing `end_capture` later.
        let (stream, failures) = self.scheduler.stream_and_failures(&stream_id);
        if let Err(err) = stream.flush(stream_id, failures) {
            // The capture never opened: disarm retention and return to
            // `NoCapture`, so a failed `start_capture` leaves the stream fully
            // usable and re-capturable.
            stream.mem_manage.capture_end();
            stream.info_cache.capture_discard();
            stream.capturing.abort();
            return Err(err);
        }

        // Warmup is over: release the slices it retained so the recorded run
        // reuses them instead of growing the pools further.
        stream.mem_manage.capture_priming_end();
        Ok(())
    }

    fn end_capture(&mut self, stream_id: StreamId) -> Result<GraphId, ServerError> {
        // Materialize the recorded launches still queued in the scheduler.
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);

        // The capture is over even on the failure path below, so an error here
        // doesn't leave the stream stuck in capture/persistent state — and it
        // is over for a caller that does not own the window too, since that is
        // a window nobody is coming back to close. Only its owner gets a graph
        // out of it, and the errors raised inside belong to that owner rather
        // than to whoever happens to be flushing.
        let outcome = stream.capturing.end(stream_id)?;
        let recording = stream.take_recording();
        // The memory the recorded launches write. A graph that seals answers
        // for it on a failed replay; one that does not is answered for here,
        // since those launches never ran and now never will.
        let written = stream.capturing.take_recorded();
        let mut retained = stream.mem_manage.capture_end();

        // An error queued during the window (a rejected write, a failed
        // binding) means the recording is missing an operation: reject the
        // capture rather than hand back a graph that silently skips work.
        // `begin_capture` drained pre-window errors, so anything here arose
        // inside the window. Draining them here is also what keeps an
        // abandoned window from leaving its errors queued for a stream that
        // may never flush again.
        let errors = stream.flush_errors_queue(outcome.owner());
        let discarded = match outcome.is_abandoned() {
            true => Some(outcome.abandoned_error(stream_id, errors)),
            false => (!errors.is_empty()).then(|| ServerError::ServerUnhealthy {
                errors,
                backtrace: BackTrace::capture(),
            }),
        };
        if let Some(err) = discarded {
            stream.info_cache.capture_discard();
            // The recording is thrown away, so the launches in it never run:
            // every buffer they were given is left as it was. The caller gets
            // the error below; the taint is what makes a read of one of those
            // buffers fail on some other stream, which heard nothing.
            self.scheduler.taint(err.clone(), written.iter());
            return Err(err);
        }

        let id = GraphId::new();
        // Seal the info-cache entries this capture pinned under the graph's
        // id, so `graph_destroy` can release them later.
        stream.info_cache.capture_commit(id);
        retained.extend(recording.uniform_pins);
        self.graphs.insert(
            id,
            WgpuGraph {
                tasks: recording.tasks,
                _retained: retained,
                _shared: recording.shared,
                written,
            },
        );
        Ok(id)
    }

    fn replay(&mut self, graph: GraphId, stream_id: StreamId) {
        // Order the replay after previously queued work on this stream.
        self.scheduler.execute_streams(vec![stream_id]);

        // Fire-and-forget like `launch`: on failure, push the error onto the
        // stream's queue so it surfaces on the next flush/sync rather than
        // blocking the caller here.
        // Nothing to name here: the graph is gone, and with it the record of
        // which buffers its launches would have written.
        let Some(wgpu_graph) = self.graphs.get(&graph) else {
            let stream = self.scheduler.stream(&stream_id);
            stream.errors.push(
                stream_id,
                ServerError::graph_state("replay was given an unknown or already-destroyed graph"),
            );
            return;
        };
        let recording = self
            .scheduler
            .stream(&stream_id)
            .reject_while_recording("replay");
        if let Err(err) = recording {
            // None of the recorded launches run, so every buffer the graph
            // writes is left as it was.
            self.scheduler.taint(err.clone(), wgpu_graph.written.iter());
            self.scheduler
                .stream(&stream_id)
                .errors
                .push(stream_id, err);
            return;
        }
        let (stream, failures) = self.scheduler.stream_and_failures(&stream_id);
        stream.replay_graph(wgpu_graph, failures);
    }

    fn graph_destroy(&mut self, graph: GraphId, stream_id: StreamId) {
        // No-op for an unknown id (e.g. a double release). The graph is held
        // until the end of this function, so its pins outlive the flush below.
        let Some(wgpu_graph) = self.graphs.remove(&graph) else {
            return;
        };
        let (stream, failures) = self.scheduler.stream_and_failures(&stream_id);
        // Submit any replay still sitting in the encoder before the pins drop:
        // a `queue.write_buffer` onto a reclaimed slice runs at the *next*
        // submit, ahead of everything already in the encoder, so it would reach
        // the GPU before the still-unsubmitted replay that reads it. The `Write`
        // path flushes on its own account, so this covers the writes that do
        // not — the uniform uploads in `create_uniform`/`info_uniform`. Once the
        // replay is submitted, queue ordering makes releasing the slices safe
        // with no host sync, unlike CUDA.
        stream.submit(failures);
        // Release the info-cache entries this graph pinned; entries no other
        // live graph still pins are dropped, freeing their buffers.
        stream.info_cache.graph_release(graph);
        drop(wgpu_graph);
    }
}

pub(crate) fn contiguous_strides(shape: &Shape) -> Strides {
    let rank = shape.len();
    let mut strides = strides![1; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
