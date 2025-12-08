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
    server::{
        Allocation, AllocationDescriptor, Binding, Bindings, ComputeServer, CopyDescriptor,
        ExecutionError, IoError, LaunchError, ProfileError, ProfilingToken, ServerCommunication,
        ServerUtilities,
    },
};
use cubecl_runtime::{
    compiler::CubeTask,
    config::GlobalConfig,
    id::KernelId,
    logging::ServerLogger,
    memory_management::{MemoryAllocationMode, MemoryDeviceProperties},
    storage::{BindingResource, BytesStorage, ComputeStorage},
    stream::scheduler::{SchedulerMultiStream, SchedulerMultiStreamOptions, SchedulerStrategy},
};
use std::{collections::HashMap, sync::Arc};

#[derive(Debug)]
pub struct CpuServer {
    scheduler: SchedulerMultiStream<ScheduledCpuBackend>,
    utilities: Arc<ServerUtilities<CpuServer>>,
    compilation_cache: HashMap<KernelId, CpuKernel>,
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
        }
    }

    fn prepare_bindings(&mut self, bindings: Bindings) -> BindingsResource {
        // Store all the resources we'll be using. This could be eliminated if
        // there was a way to tie the lifetime of the resource to the memory handle.
        let resources = bindings
            .buffers
            .into_iter()
            .map(|b| {
                let stream = self.scheduler.stream(&b.stream);
                stream
                    .memory_management
                    .get_resource(b.memory, b.offset_start, b.offset_end)
                    .unwrap()
            })
            .collect::<Vec<_>>();

        BindingsResource {
            resources,
            metadata: bindings.metadata,
            scalars: bindings.scalars,
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
                let handle = stream
                    .memory_management
                    .get_resource(binding.memory, binding.offset_start, binding.offset_end)
                    .expect("Failed to find resource");
                stream.flush();

                let bytes = handle.read();
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
}

impl ComputeServer for CpuServer {
    type Kernel = Box<dyn CubeTask<CpuCompiler>>;
    type Storage = BytesStorage;
    type Info = ();

    fn logger(&self) -> Arc<ServerLogger> {
        self.scheduler.logger.clone()
    }

    fn staging(&mut self, _sizes: &[usize], _stream_id: StreamId) -> Result<Vec<Bytes>, IoError> {
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        })
    }

    fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }

    fn create(
        &mut self,
        descriptors: Vec<AllocationDescriptor<'_>>,
        stream_id: StreamId,
    ) -> Result<Vec<Allocation>, IoError> {
        let stream = self.scheduler.stream(&stream_id);
        stream.create(descriptors, stream_id)
    }

    fn read<'a>(
        &mut self,
        descriptors: Vec<CopyDescriptor<'a>>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, IoError>> {
        let mut streams = vec![stream_id];
        let mut results = Vec::with_capacity(descriptors.len());
        let mut ressources = Vec::with_capacity(descriptors.len());

        // Since we do a zero-copy read, we can collect bytes before synching the streams.
        for desc in descriptors {
            if !streams.contains(&desc.binding.stream) {
                streams.push(desc.binding.stream);
            }
            let stream = self.scheduler.stream(&stream_id);
            let result = stream.read_async(desc);
            results.push(result);
        }

        self.scheduler.execute_streams(streams);

        Box::pin(async move {
            for result in results {
                match result.await {
                    Ok(val) => ressources.push(val),
                    Err(err) => return Err(err),
                }
            }

            Ok(ressources)
        })
    }

    fn write(
        &mut self,
        descriptors: Vec<(CopyDescriptor<'_>, Bytes)>,
        stream_id: StreamId,
    ) -> Result<(), IoError> {
        for (desc, data) in descriptors {
            let stream = self.scheduler.stream(&desc.binding.stream);
            let resource = stream
                .memory_management
                .get_resource(
                    desc.binding.memory,
                    desc.binding.offset_start,
                    desc.binding.offset_end,
                )
                .ok_or_else(|| IoError::InvalidHandle {
                    backtrace: BackTrace::capture(),
                })?;
            let task = ScheduleTask::Write {
                data,
                buffer: resource,
            };

            self.scheduler.register(stream_id, task, [].into_iter());
        }

        Ok(())
    }

    fn memory_usage(&mut self, stream_id: StreamId) -> MemoryUsage {
        let stream = self.scheduler.stream(&stream_id);
        stream.memory_management.memory_usage()
    }

    fn memory_cleanup(&mut self, stream_id: StreamId) {
        let stream = self.scheduler.stream(&stream_id);
        stream.memory_management.cleanup(true)
    }

    unsafe fn launch(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: Bindings,
        kind: ExecutionMode,
        stream_id: StreamId,
    ) -> Result<(), LaunchError> {
        let buffers = bindings.buffers.clone();
        let bindings = self.prepare_bindings(bindings);
        let task = self.prepare_task(kernel, count, bindings, kind)?;

        self.scheduler.register(stream_id, task, buffers.iter());

        Ok(())
    }

    fn flush(&mut self, stream_id: StreamId) {
        let stream = self.scheduler.stream(&stream_id);
        stream.flush();
    }

    fn sync(&mut self, stream_id: StreamId) -> DynFut<Result<(), ExecutionError>> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.flush();

        Box::pin(async move { Ok(()) })
    }

    fn start_profile(&mut self, stream_id: StreamId) -> ProfilingToken {
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
    ) -> BindingResource<<Self::Storage as ComputeStorage>::Resource> {
        let mut streams = vec![stream_id];
        if binding.stream != stream_id {
            streams.push(binding.stream);
        }
        self.scheduler.execute_streams(streams);

        let stream = self.scheduler.stream(&binding.stream);
        let resource = stream
            .memory_management
            .get_resource(
                binding.memory.clone(),
                binding.offset_start,
                binding.offset_end,
            )
            .expect("Can't find resource");

        BindingResource::new(binding, resource)
    }

    fn allocation_mode(&mut self, mode: MemoryAllocationMode, stream_id: StreamId) {
        let stream = self.scheduler.stream(&stream_id);
        stream.allocation_mode(mode);
    }
}

impl ServerCommunication for CpuServer {
    const SERVER_COMM_ENABLED: bool = false;
}

pub(crate) fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let rank = shape.len();
    let mut strides = vec![1; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
