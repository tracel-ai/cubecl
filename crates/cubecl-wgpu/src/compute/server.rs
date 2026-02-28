use super::storage::{WgpuResource, WgpuStorage};
use crate::schedule::{BindingsResource, ScheduleTask, ScheduledWgpuBackend};
use crate::{AutoCompiler, AutoRepresentation};
use alloc::sync::Arc;
use cubecl_common::{
    backtrace::BackTrace,
    bytes::Bytes,
    profile::{ProfileDuration, TimingMethod},
    stream_id::StreamId,
};
use cubecl_core::server::{Binding, HandleId};
use cubecl_core::zspace::Shape;
use cubecl_core::{
    MemoryConfiguration, WgpuCompilationOptions,
    future::DynFut,
    prelude::*,
    server::{
        CopyDescriptor, IoError, KernelArguments, LaunchError, ProfileError, ProfilingToken,
        ResourceLimitError, ServerCommunication, ServerError, ServerUtilities,
    },
    zspace::{Strides, strides},
};
#[cfg(feature = "spirv")]
use cubecl_core::{cache::CacheOption, compilation_cache::CompilationCache, hash::StableHash};
use cubecl_ir::MemoryDeviceProperties;
use cubecl_runtime::allocator::ContiguousMemoryLayoutPolicy;
use cubecl_runtime::memory_management::MemoryUsage;
use cubecl_runtime::{
    compiler::CubeTask,
    config::GlobalConfig,
    logging::ServerLogger,
    memory_management::MemoryAllocationMode,
    server::ComputeServer,
    storage::ManagedResource,
    stream::scheduler::{SchedulerMultiStream, SchedulerMultiStreamOptions, SchedulerStrategy},
    validation::{validate_cube_dim, validate_units},
};
use hashbrown::HashMap;
use wgpu::ComputePipeline;

/// Wgpu compute server.
#[derive(Debug)]
pub struct WgpuServer {
    pub(crate) device: wgpu::Device,
    pipelines: HashMap<KernelId, Arc<ComputePipeline>>,
    scheduler: SchedulerMultiStream<ScheduledWgpuBackend>,
    #[cfg(feature = "spirv")]
    pub(crate) spirv_cache:
        Option<CompilationCache<(u64, StableHash), cubecl_spirv::SpirvCacheEntry>>,
    pub compilation_options: WgpuCompilationOptions,
    pub(crate) backend: wgpu::Backend,
    pub(crate) utilities: Arc<ServerUtilities<Self>>,
}

impl ServerCommunication for WgpuServer {
    const SERVER_COMM_ENABLED: bool = false;
}

impl WgpuServer {
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
        let backend_scheduler = ScheduledWgpuBackend::new(
            device.clone(),
            queue.clone(),
            memory_properties,
            memory_config,
            timing_method,
            tasks_max,
            utilities.logger.clone(),
        );

        let config = GlobalConfig::get();
        let max_streams = config.streaming.max_streams;

        Self {
            compilation_options,
            device,
            pipelines: HashMap::new(),
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
            spirv_cache: {
                let config = cubecl_runtime::config::GlobalConfig::get();
                if let Some(cache) = &config.compilation.cache {
                    let root = cache.root();
                    Some(CompilationCache::new(
                        "spirv",
                        CacheOption::default().name("vulkan").root(root),
                    ))
                } else {
                    None
                }
            },
            backend,
            utilities: Arc::new(utilities),
        }
    }

    fn prepare_bindings(&mut self, bindings: KernelArguments) -> Result<BindingsResource, IoError> {
        // Store all the resources we'll be using. This could be eliminated if
        // there was a way to tie the lifetime of the resource to the memory handle.
        let mut resources = Vec::with_capacity(bindings.buffers.len());

        for b in bindings.buffers.iter() {
            let stream = self.scheduler.stream(&b.stream);
            let resource = stream.mem_manage.get_resource(b.clone())?.0;
            resources.push(resource);
        }

        Ok(BindingsResource {
            resources,
            metadata: bindings.metadata,
            scalars: bindings.scalars,
        })
    }

    fn pipeline(
        &mut self,
        kernel: <Self as ComputeServer>::Kernel,
        bindings: &KernelArguments,
        mode: ExecutionMode,
    ) -> Result<Arc<ComputePipeline>, LaunchError> {
        let mut kernel_id = kernel.id();
        kernel_id.mode(mode);

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

        let mut compiler = compiler(self.backend);
        let mut compiled = compiler.compile(self, kernel, mode)?;

        if self.scheduler.logger.compilation_activated() {
            compiled.debug_info = Some(DebugInformation::new(
                compiler.lang_tag(),
                kernel_id.clone(),
            ));
        }
        self.scheduler.logger.log_compilation(&compiled);

        self.validate_shared(&compiled.repr)?;

        // /!\ Do not delete the following commented code.
        // This is useful while working on the metal compiler.
        // Also the errors are printed nicely which is not the case when this is the runtime
        // that does it.
        // println!("SOURCE:\n{}", compile.source);
        // {
        //     // Write shader in metal file then compile it for error
        //     std::fs::write("shader.metal", &compile.source).expect("should write to file");
        //     let _status = std::process::Command::new("xcrun")
        //         .args(vec![
        //             "-sdk",
        //             "macosx",
        //             "metal",
        //             "-o",
        //             "shader.ir",
        //             "-c",
        //             "shader.metal",
        //         ])
        //         .status()
        //         .expect("should launch the command");
        //     // std::process::exit(status.code().unwrap());
        // }
        let repr = compiled.repr.as_ref().map(|it| it.as_ref());
        let module = self.create_module(&compiled.entrypoint_name, repr, &compiled.source, mode)?;
        let pipeline = self.create_pipeline(&compiled.entrypoint_name, repr, module, bindings);
        self.pipelines.insert(kernel_id.clone(), pipeline.clone());

        #[cfg(feature = "spirv")]
        if let Some(Err(key)) = cached
            && let Some(crate::AutoRepresentation::SpirV(kernel)) = compiled.repr
        {
            let cache = self.spirv_cache.as_mut().unwrap();
            let result = cache.insert(
                key,
                cubecl_spirv::SpirvCacheEntry::new(compiled.entrypoint_name, kernel),
            );
            if let Err(err) = result {
                log::warn!("Unable to save the SPIR-V {err:?}");
            }
        }

        Ok(pipeline)
    }

    fn validate_shared(&self, repr: &Option<crate::AutoRepresentation>) -> Result<(), LaunchError> {
        let shared_bytes = repr.as_ref().map(|repr| match repr {
            AutoRepresentation::Wgsl(repr) => repr.shared_memory_bytes(),
            #[cfg(feature = "msl")]
            AutoRepresentation::Msl(repr) => repr.shared_memory_size(),
            #[cfg(feature = "spirv")]
            AutoRepresentation::SpirV(repr) => repr.shared_size,
        });
        let max_smem = self.utilities.properties.hardware.max_shared_memory_size;
        if let Some(shared_bytes) = shared_bytes
            && shared_bytes > max_smem
        {
            Err(ResourceLimitError::SharedMemory {
                requested: shared_bytes,
                max: max_smem,
                backtrace: BackTrace::capture(),
            }
            .into())
        } else {
            Ok(())
        }
    }
}

impl ComputeServer for WgpuServer {
    type Kernel = Box<dyn CubeTask<AutoCompiler>>;
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

    fn initialize_bindings(&mut self, handles: Vec<Binding>, stream_id: StreamId) {
        let stream = self.scheduler.stream(&stream_id);
        if !stream.is_healthy() {
            stream.error(ServerError::ServerUnHealthy {
                reason: "Can't create a tensor, since the stream isn't in an healthy state"
                    .to_string(),
                backtrace: BackTrace::capture(),
            });
            return;
        }

        let mut memory_size = 0;

        for handle in handles.iter() {
            memory_size += handle.size();
        }

        let memory = stream.empty(memory_size).unwrap();
        let slots = memory.partition(memory_size, &handles, 0, stream_id);
        stream.bind(slots, handles);
    }

    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>> {
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

            if !stream.is_healthy() {
                return Box::pin(async move {
                    Err(ServerError::ServerUnHealthy {
                        reason: "Stream is in an invalid state.".to_string(),
                        backtrace: BackTrace::capture(),
                    })
                });
            }

            let resource = match stream.mem_manage.get_resource(desc.handle) {
                Ok(val) => val.0,
                Err(err) => return Box::pin(async move { Err(err.into()) }),
            };
            resources.push((resource, desc.shape, desc.elem_size));
        }

        self.scheduler.execute_streams(streams);
        let stream = self.scheduler.stream(&stream_id);
        stream.read_resources(resources)
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
            let resource = match stream.mem_manage.get_resource(desc.handle.clone()) {
                Ok(r) => r.0,
                Err(err) => {
                    stream.error(ServerError::Io(err));
                    return;
                }
            };
            let task = ScheduleTask::Write {
                data,
                buffer: resource,
            };

            self.scheduler.register(stream_id, task, [].into_iter());
        }
    }

    fn get_resource(
        &mut self,
        binding: Binding,
        stream_id: StreamId,
    ) -> Result<ManagedResource<WgpuResource>, ServerError> {
        let mut streams = vec![stream_id];
        if binding.stream != stream_id {
            streams.push(binding.stream);
        }
        self.scheduler.execute_streams(streams);
        let stream = self.scheduler.stream(&binding.stream);
        let (resource, handle) = stream.mem_manage.get_resource(binding.clone())?;

        Ok(ManagedResource::new(handle, resource))
    }

    unsafe fn launch(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        args: KernelArguments,
        mode: ExecutionMode,
        stream_id: StreamId,
    ) {
        let pipeline = match self.pipeline(kernel, &args, mode) {
            Ok(val) => val,
            Err(err) => {
                // We make the stream that would execute the kernel in error.
                let stream = self.scheduler.stream(&stream_id);
                stream.errors.push(ServerError::Launch(err));
                return;
            }
        };

        let buffers = args.buffers.clone();
        let resources = match self.prepare_bindings(args) {
            Ok(val) => val,
            Err(err) => {
                // We make the stream that would execute the kernel in error.
                let stream = self.scheduler.stream(&stream_id);
                stream.errors.push(ServerError::Io(err));
                return;
            }
        };
        let task = ScheduleTask::Execute {
            pipeline,
            count,
            resources,
        };

        self.scheduler.register(stream_id, task, buffers.iter());
    }

    fn flush(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        if !stream.is_healthy() {
            return Err(ServerError::ServerUnHealthy {
                reason: "Server is not healthy, can't flush".to_string(),
                backtrace: BackTrace::capture(),
            });
        }
        stream.flush();
        Ok(())
    }

    /// Returns the total time of GPU work this sync completes.
    fn sync(&mut self, stream_id: StreamId) -> DynFut<Result<(), ServerError>> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.sync()
    }

    fn start_profile(&mut self, stream_id: StreamId) -> Result<ProfilingToken, ServerError> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);

        if !stream.is_healthy() {
            return Err(ServerError::ServerUnHealthy {
                reason: "Server is not healthy, can't flush".to_string(),
                backtrace: BackTrace::capture(),
            });
        }

        Ok(stream.start_profile())
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

    fn memory_usage(&mut self, stream_id: StreamId) -> Result<MemoryUsage, ServerError> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        Ok(stream.mem_manage.memory_usage())
    }

    fn memory_cleanup(&mut self, stream_id: StreamId) {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.mem_manage.memory_cleanup(true);
    }

    fn allocation_mode(&mut self, mode: MemoryAllocationMode, stream_id: StreamId) {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.mem_manage.mode(mode);
    }

    fn flush_errors(&mut self, stream_id: StreamId) -> Vec<ServerError> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.flush();
        let errors = core::mem::take(&mut stream.errors);
        self.memory_cleanup(stream_id);
        errors
    }

    fn free(&mut self, handle: HandleId, stream_id: StreamId) {
        self.scheduler
            .register(stream_id, ScheduleTask::Free { handle }, [].into_iter());
    }
}

fn compiler(backend: wgpu::Backend) -> AutoCompiler {
    match backend {
        #[cfg(feature = "spirv")]
        wgpu::Backend::Vulkan => AutoCompiler::SpirV(Default::default()),
        #[cfg(feature = "msl")]
        wgpu::Backend::Metal => AutoCompiler::Msl(Default::default()),
        _ => AutoCompiler::Wgsl(Default::default()),
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
