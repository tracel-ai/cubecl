use cubecl_llvm::PlironOptions;

use crate::{
    CpuCompiler,
    compute::{
        cpu_kernel::CpuKernel,
        schedule::{BindingsResource, ScheduleTask, ScheduledCpuBackend},
    },
};
use cubecl_common::{bytes::Bytes, profile::ProfileDuration};
use cubecl_core::{
    CompilationError, CubeCount, MemoryConfiguration, MemoryUsage,
    ir::MemoryDeviceProperties,
    server::{
        BufferBinding, ComputeServer, CopyDescriptor, IoError, KernelArguments, KernelResource,
        LaunchError, ProfileError, ProfilingToken, ServerCommunication, ServerError,
        ServerUtilities,
    },
    zspace::{Shape, Strides, strides},
};
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::future::DynFut;
use cubecl_environment::stream::StreamId;
use cubecl_runtime::{
    allocator::ContiguousMemoryLayoutPolicy,
    compiler::CubeTask,
    config::{CubeClRuntimeConfig, RuntimeConfig},
    dry_run::LaunchMode,
    id::KernelId,
    logging::ServerLogger,
    memory_management::{ManagedMemoryHandle, MemoryAllocationMode},
    storage::{BytesStorage, ComputeStorage, ManagedResource},
    stream::WriteScoped,
    stream::scheduler::{SchedulerMultiStream, SchedulerMultiStreamOptions, SchedulerStrategy},
};
use std::{collections::HashMap, sync::Arc};

#[derive(Debug)]
pub struct CpuServer {
    scheduler: SchedulerMultiStream<ScheduledCpuBackend>,
    utilities: Arc<ServerUtilities<CpuServer>>,
    compilation_cache: HashMap<KernelId, CpuKernel>,
    // A buffer that can be used to store stream id without extra allocations.
    streams_pool: Vec<StreamId>,
}

impl WriteScoped for CpuServer {
    type Streams = SchedulerMultiStream<ScheduledCpuBackend>;

    fn write_streams(&mut self) -> &mut Self::Streams {
        &mut self.scheduler
    }
}

impl CpuServer {
    pub fn new(
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        utilities: Arc<ServerUtilities<CpuServer>>,
    ) -> Self {
        let backend =
            ScheduledCpuBackend::new(memory_properties, memory_config, utilities.logger.clone());
        let config = CubeClRuntimeConfig::get();
        let max_streams = config.streaming.max_streams;

        let scheduler = SchedulerMultiStream::new(
            utilities.logger.clone(),
            backend,
            SchedulerMultiStreamOptions {
                max_streams,
                max_tasks: 8,
                strategy: SchedulerStrategy::Interleave,
            },
        );

        Self {
            scheduler,
            utilities,
            compilation_cache: HashMap::new(),
            streams_pool: Vec::new(),
        }
    }

    fn prepare_bindings(&mut self, bindings: KernelArguments) -> BindingsResource {
        // Store all the resources we'll be using. This could be eliminated if
        // there was a way to tie the lifetime of the resource to the memory handle.
        let resources = bindings
            .resources
            .into_iter()
            .filter_map(|binding| {
                let KernelResource::Buffer(binding) = binding else {
                    return None;
                };
                let stream = self.scheduler.stream(&binding.stream);
                let memory = binding.memory.clone();
                let resource = stream
                    .memory_management
                    .get_resource(binding.memory, binding.offset_start, binding.offset_end)
                    .unwrap();
                Some(ManagedResource::new(memory, resource))
            })
            .collect::<Vec<_>>();

        BindingsResource {
            resources,
            info: bindings.info,
        }
    }

    fn prepare_task(
        &mut self,
        kernel: Box<dyn CubeTask<CpuCompiler>>,
        count: CubeCount,
        bindings: BindingsResource,
        stream_id: StreamId,
    ) -> Result<ScheduleTask, CompilationError> {
        let cube_count = match count {
            CubeCount::Static(x, y, z) => [x, y, z],
            CubeCount::Dynamic(binding) => {
                let stream = self.scheduler.stream(&binding.stream);
                let resource = stream
                    .memory_management
                    .get_resource(binding.memory, binding.offset_start, binding.offset_end)
                    .unwrap();

                stream.submit();

                let bytes = resource.read();
                let x = u32::from_ne_bytes(bytes[0..4].try_into().unwrap());
                let y = u32::from_ne_bytes(bytes[4..8].try_into().unwrap());
                let z = u32::from_ne_bytes(bytes[8..12].try_into().unwrap());
                [x, y, z]
            }
        };

        self.prepare_task_inner(kernel, cube_count, bindings, stream_id)
    }

    /// Compile and cache `kernel` without scheduling anything — everything a
    /// skipped launch owes the caches, touching no buffer.
    fn compile_only(
        &mut self,
        kernel: Box<dyn CubeTask<CpuCompiler>>,
    ) -> Result<(), CompilationError> {
        let kernel_id = kernel.id();
        if self.compilation_cache.contains_key(&kernel_id) {
            return Ok(());
        }
        let definition = kernel.define();
        let compiled = kernel.compile(definition, &mut Default::default(), &PlironOptions)?;
        self.compilation_cache
            .insert(kernel_id, CpuKernel::new(compiled));
        Ok(())
    }

    fn prepare_task_inner(
        &mut self,
        kernel: Box<dyn CubeTask<CpuCompiler>>,
        cube_count: [u32; 3],
        bindings: BindingsResource,
        stream_id: StreamId,
    ) -> Result<ScheduleTask, CompilationError> {
        let kernel_id = kernel.id();
        self.compile_only(kernel)?;
        let kernel = self
            .compilation_cache
            .get_mut(&kernel_id)
            .expect("just compiled");

        let cube_dim = kernel.mlir.cube_dim;

        let mlir_engine = kernel.mlir.repr.clone().unwrap();

        let task = ScheduleTask::Execute {
            stream_id,
            pliron_engine: mlir_engine,
            bindings,
            cube_dim,
            cube_count,
        };

        Ok(task)
    }

    pub(crate) fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }
}

impl ComputeServer for CpuServer {
    type Kernel = Box<dyn CubeTask<CpuCompiler>>;
    type Storage = BytesStorage;
    type MemoryLayoutPolicy = ContiguousMemoryLayoutPolicy;
    type Info = ();

    fn logger(&self) -> Arc<ServerLogger> {
        self.scheduler.logger.clone()
    }

    fn staging(
        &mut self,
        _sizes: &[usize],
        _stream_id: StreamId,
    ) -> Result<Vec<Bytes>, ServerError> {
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }

    fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }

    fn initialize_memory(&mut self, memory: ManagedMemoryHandle, size: u64, stream_id: StreamId) {
        let (stream, failures) = self.scheduler.stream_and_failures(&stream_id);
        let reserved = stream.empty(size, failures).unwrap();
        stream.bind(reserved, memory, failures);
    }

    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        // Buffers another stream wrote are only as good as the work that wrote
        // them; see `StreamPool::ensure_written`.
        if let Err(err) = self
            .scheduler
            .ensure_written(descriptors.iter().map(|d| &d.handle))
        {
            return Box::pin(async move { Err(err) });
        }

        let mut streams = vec![stream_id];
        let mut results = Vec::with_capacity(descriptors.len());
        let mut resources = Vec::with_capacity(descriptors.len());

        // Since we do a zero-copy read, we can collect bytes before synching the streams.
        for desc in descriptors {
            if !streams.contains(&desc.handle.stream) {
                streams.push(desc.handle.stream);
            }
            // The resource lives in the memory management of the stream that
            // owns the handle, which is not always the reader's.
            let stream = self.scheduler.stream(&desc.handle.stream);
            let result = stream.read_async(desc);
            results.push(result);
        }

        self.scheduler.execute_streams(streams);

        // The reader's own errors, on the way out: a launch that failed here
        // never wrote the buffers either, so the bytes below are stale.
        if let Err(err) = self.scheduler.stream(&stream_id).flush(stream_id) {
            return Box::pin(async move { Err(err) });
        }

        Box::pin(async move {
            for result in results {
                match result.await {
                    Ok(val) => resources.push(val),
                    Err(err) => return Err(err.into()),
                }
            }

            Ok(resources)
        })
    }

    fn write(&mut self, descriptors: Vec<(CopyDescriptor, Bytes)>, stream_id: StreamId) {
        // No health gate, as on every other backend: a queued error is the
        // caller's to surface at its next flush, and refusing the write here
        // would leave the buffer unwritten for a caller whose flush has already
        // reported and cleared that error.
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
                    let memory = desc.handle.memory.clone();
                    let stream = server.scheduler.stream(&owner);
                    let resource = stream.get_resource(desc.handle).map_err(ServerError::Io)?;
                    let task = ScheduleTask::Write {
                        data,
                        buffer: ManagedResource::new(memory, resource),
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

    fn memory_usage(&mut self, stream_id: StreamId) -> Result<MemoryUsage, ServerError> {
        let stream = self.scheduler.stream(&stream_id);
        Ok(stream.memory_management.memory_usage())
    }

    fn memory_report(
        &mut self,
        stream_id: StreamId,
    ) -> Result<cubecl_runtime::memory_management::MemoryReport, ServerError> {
        let stream = self.scheduler.stream(&stream_id);
        Ok(stream.memory_management.memory_report())
    }

    fn stream_ids(&self) -> Vec<StreamId> {
        self.scheduler.stream_ids().collect()
    }

    fn memory_cleanup(&mut self, stream_id: StreamId) {
        let (stream, failures) = self.scheduler.stream_and_failures(&stream_id);
        stream.memory_management.cleanup(true, failures)
    }

    unsafe fn launch(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: KernelArguments,
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
            bindings,
            |bindings, written| {
                if !launch_mode.is_skipped() {
                    written.extend(bindings.buffers_written().cloned());
                }
            },
            |server, bindings, _| {
                // A skipped launch stops here, after compilation and before
                // anything that touches a buffer: resolving resources or
                // reading a dynamic cube count would materialize memory a dry
                // run exists to leave unmapped. It registers no stream
                // dependency either, which is correct rather than an
                // oversight — nothing is scheduled, so there is no work for a
                // later stream to order against.
                if launch_mode.is_skipped() {
                    return server
                        .compile_only(kernel)
                        .map(|_| ())
                        .map_err(|err| ServerError::Launch(LaunchError::CompilationError(err)));
                }

                server.streams_pool.clear();
                bindings
                    .resources
                    .iter()
                    .filter_map(|b| {
                        let KernelResource::Buffer(b) = b else {
                            return None;
                        };
                        Some(b)
                    })
                    .for_each(|b| server.streams_pool.push(b.stream));
                let bindings = server.prepare_bindings(bindings);
                let task = server
                    .prepare_task(kernel, count, bindings, stream_id)
                    .map_err(|err| ServerError::Launch(LaunchError::CompilationError(err)))?;

                server
                    .scheduler
                    .register(stream_id, task, &server.streams_pool);
                Ok(())
            },
        );
    }

    fn flush(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.flush(stream_id)
    }

    fn sync(&mut self, stream_id: StreamId) -> DynFut<Result<(), ServerError>> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        let result = stream.flush(stream_id);

        Box::pin(async move { result })
    }

    fn start_profile(&mut self, stream_id: StreamId) -> Result<ProfilingToken, ServerError> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.start_profile(stream_id)
    }

    fn end_profile(
        &mut self,
        stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.end_profile(token, stream_id)
    }

    fn get_resource(
        &mut self,
        binding: BufferBinding,
        stream_id: StreamId,
    ) -> Result<ManagedResource<<Self::Storage as ComputeStorage>::Resource>, ServerError> {
        let mut streams = vec![stream_id];
        if binding.stream != stream_id {
            streams.push(binding.stream);
        }
        self.scheduler.execute_streams(streams);

        let stream = self.scheduler.stream(&binding.stream);
        let memory = binding.memory.clone();
        let resource = stream.get_resource(binding)?;

        Ok(ManagedResource::new(memory, resource))
    }

    fn allocation_mode(&mut self, mode: MemoryAllocationMode, stream_id: StreamId) {
        let stream = self.scheduler.stream(&stream_id);
        stream.allocation_mode(mode);
    }
}

impl ServerCommunication for CpuServer {
    const SERVER_COMM_ENABLED: bool = false;
}

pub(crate) fn contiguous_strides(shape: &Shape) -> Strides {
    let rank = shape.len();
    let mut strides = strides![1; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
