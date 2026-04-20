use crate::{
    CpuCompiler,
    compiler::MlirCompilerOptions,
    compute::{
        runner::CpuKernel,
        schedule::{BindingsResource, ScheduleTask, ScheduledCpuBackend},
    },
};
use cubecl_common::{
    backtrace::BackTrace, bytes::Bytes, profile::ProfileDuration, stream_id::StreamId,
};
use cubecl_core::{
    CompilationError, CubeCount, ExecutionMode, MemoryConfiguration, MemoryUsage,
    future::DynFut,
    ir::MemoryDeviceProperties,
    server::{
        Binding, ComputeServer, CopyDescriptor, IoError, KernelArguments, ProfileError,
        ProfilingToken, ServerCommunication, ServerError, ServerUtilities,
    },
    zspace::{Shape, Strides, strides},
};
use cubecl_runtime::server::LaunchError;
use cubecl_runtime::{
    allocator::ContiguousMemoryLayoutPolicy,
    compiler::CubeTask,
    config::GlobalConfig,
    id::KernelId,
    logging::ServerLogger,
    memory_management::{ManagedMemoryHandle, MemoryAllocationMode},
    storage::{BytesStorage, ComputeStorage, ManagedResource},
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

impl CpuServer {
    pub fn new(
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        utilities: Arc<ServerUtilities<CpuServer>>,
    ) -> Self {
        let backend =
            ScheduledCpuBackend::new(memory_properties, memory_config, utilities.logger.clone());
        let config = GlobalConfig::get();
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
            .buffers
            .into_iter()
            .map(|binding| {
                let stream = self.scheduler.stream(&binding.stream);
                stream
                    .memory_management
                    .get_resource(binding.memory, binding.offset_start, binding.offset_end)
                    .unwrap()
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
        kind: ExecutionMode,
    ) -> Result<ScheduleTask, CompilationError> {
        let cube_count = match count {
            CubeCount::Static(x, y, z) => [x, y, z],
            CubeCount::Dynamic(binding) => {
                let stream = self.scheduler.stream(&binding.stream);
                let resource = stream
                    .memory_management
                    .get_resource(binding.memory, binding.offset_start, binding.offset_end)
                    .unwrap();

                let _ = stream
                    .flush(cubecl_core::server::StreamErrorMode {
                        ignore: true,
                        flush: false,
                    })
                    .ok();

                let bytes = resource.read();
                let x = u32::from_ne_bytes(bytes[0..4].try_into().unwrap());
                let y = u32::from_ne_bytes(bytes[4..8].try_into().unwrap());
                let z = u32::from_ne_bytes(bytes[8..12].try_into().unwrap());
                [x, y, z]
            }
        };

        self.prepare_task_inner(kernel, cube_count, bindings, kind)
    }

    fn prepare_task_inner(
        &mut self,
        kernel: Box<dyn CubeTask<CpuCompiler>>,
        cube_count: [u32; 3],
        bindings: BindingsResource,
        kind: ExecutionMode,
    ) -> Result<ScheduleTask, CompilationError> {
        let kernel_id = kernel.id();
        let kernel = if let Some(kernel) = self.compilation_cache.get(&kernel_id) {
            kernel
        } else {
            let kernel = kernel.compile(
                &mut Default::default(),
                &MlirCompilerOptions::default(),
                kind,
                kernel.address_type(),
            )?;
            self.compilation_cache
                .insert(kernel_id.clone(), CpuKernel::new(kernel));
            self.compilation_cache
                .get_mut(&kernel_id)
                .expect("Just inserted")
        };

        let cube_dim = kernel.mlir.cube_dim;

        let mlir_engine = kernel.mlir.repr.clone().unwrap();

        let task = ScheduleTask::Execute {
            mlir_engine,
            bindings,
            kind,
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
        let stream = self.scheduler.stream(&stream_id);
        let reserved = stream.empty(size).unwrap();
        stream.bind(reserved, memory);
    }

    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        let mut streams = vec![stream_id];
        let mut results = Vec::with_capacity(descriptors.len());
        let mut resources = Vec::with_capacity(descriptors.len());

        // Since we do a zero-copy read, we can collect bytes before synching the streams.
        for desc in descriptors {
            if !streams.contains(&desc.handle.stream) {
                streams.push(desc.handle.stream);
            }
            let stream = self.scheduler.stream(&stream_id);
            let result = stream.read_async(desc);
            results.push(result);
        }

        self.scheduler.execute_streams(streams);

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
        for (desc, data) in descriptors {
            let stream = self.scheduler.stream(&desc.handle.stream);

            if contiguous_strides(&desc.shape) != desc.strides {
                stream.error(ServerError::Io(IoError::UnsupportedStrides {
                    backtrace: BackTrace::capture(),
                }));
                return;
            }

            if !stream.is_healthy() {
                return;
            }

            let resource = match stream.get_resource(desc.handle.clone()) {
                Ok(r) => r,
                Err(err) => {
                    stream.error(ServerError::Io(err));
                    return;
                }
            };
            let task = ScheduleTask::Write {
                data,
                buffer: resource,
            };

            self.scheduler.register(stream_id, task, &[]);
        }
    }

    fn memory_usage(&mut self, stream_id: StreamId) -> Result<MemoryUsage, ServerError> {
        let stream = self.scheduler.stream(&stream_id);
        Ok(stream.memory_management.memory_usage())
    }

    fn memory_cleanup(&mut self, stream_id: StreamId) {
        let stream = self.scheduler.stream(&stream_id);
        stream.memory_management.cleanup(true)
    }

    unsafe fn launch(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: KernelArguments,
        kind: ExecutionMode,
        stream_id: StreamId,
    ) {
        self.streams_pool.clear();
        bindings
            .buffers
            .iter()
            .for_each(|b| self.streams_pool.push(b.stream));
        let bindings = self.prepare_bindings(bindings);
        let task = match self.prepare_task(kernel, count, bindings, kind) {
            Ok(task) => task,
            Err(error) => {
                let stream = self.scheduler.stream(&stream_id);
                stream.error(LaunchError::CompilationError(error).into());
                return;
            }
        };

        self.scheduler.register(stream_id, task, &self.streams_pool);
    }

    fn flush(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.flush(cubecl_core::server::StreamErrorMode {
            ignore: false,
            flush: true,
        })
    }

    fn sync(&mut self, stream_id: StreamId) -> DynFut<Result<(), ServerError>> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        let result = stream.sync();

        Box::pin(async move { result })
    }

    fn start_profile(&mut self, stream_id: StreamId) -> Result<ProfilingToken, ServerError> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.start_profile()
    }

    fn end_profile(
        &mut self,
        stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.end_profile(token)
    }

    fn get_resource(
        &mut self,
        binding: Binding,
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
