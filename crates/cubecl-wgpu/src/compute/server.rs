use super::storage::{WgpuResource, WgpuStorage};
use crate::AutoCompiler;
use crate::schedule::{BindingsResource, ScheduleTask, ScheduledWgpuBackend};
use alloc::sync::Arc;
use cubecl_common::{
    backtrace::BackTrace,
    bytes::Bytes,
    profile::{ProfileDuration, TimingMethod},
    stream_id::StreamId,
};
use cubecl_core::{
    MemoryConfiguration, WgpuCompilationOptions,
    future::DynFut,
    prelude::*,
    server::{
        Allocation, AllocationDescriptor, Binding, Bindings, CopyDescriptor, ExecutionError,
        IoError, LaunchError, ProfileError, ProfilingToken, ServerCommunication, ServerUtilities,
    },
};
use cubecl_ir::MemoryDeviceProperties;
use cubecl_runtime::{
    compiler::{CompilationError, CubeTask},
    config::GlobalConfig,
    logging::ServerLogger,
    memory_management::{MemoryAllocationMode, offset_handles},
    server::ComputeServer,
    storage::BindingResource,
    stream::scheduler::{SchedulerMultiStream, SchedulerMultiStreamOptions, SchedulerStrategy},
};
use hashbrown::HashMap;
use wgpu::ComputePipeline;

/// Wgpu compute server.
#[derive(Debug)]
pub struct WgpuServer {
    pub(crate) device: wgpu::Device,
    pipelines: HashMap<KernelId, Arc<ComputePipeline>>,
    scheduler: SchedulerMultiStream<ScheduledWgpuBackend>,
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
            backend,
            utilities: Arc::new(utilities),
        }
    }

    fn prepare_bindings(&mut self, bindings: Bindings) -> BindingsResource {
        // Store all the resources we'll be using. This could be eliminated if
        // there was a way to tie the lifetime of the resource to the memory handle.
        let resources = bindings
            .buffers
            .iter()
            .map(|b| {
                let stream = self.scheduler.stream(&b.stream);
                stream.mem_manage.get_resource(b.clone()).unwrap()
            })
            .collect::<Vec<_>>();

        BindingsResource {
            resources,
            metadata: bindings.metadata,
            scalars: bindings.scalars,
        }
    }

    fn pipeline(
        &mut self,
        kernel: <Self as ComputeServer>::Kernel,
        mode: ExecutionMode,
    ) -> Result<Arc<ComputePipeline>, CompilationError> {
        let mut kernel_id = kernel.id();
        kernel_id.mode(mode);

        if let Some(pipeline) = self.pipelines.get(&kernel_id) {
            return Ok(pipeline.clone());
        }

        let mut compiler = compiler(self.backend);
        let mut compile = compiler.compile(self, kernel, mode)?;

        if self.scheduler.logger.compilation_activated() {
            compile.debug_info = Some(DebugInformation::new(
                compiler.lang_tag(),
                kernel_id.clone(),
            ));
        }
        self.scheduler.logger.log_compilation(&compile);
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
        let pipeline = self.create_pipeline(compile, mode)?;
        self.pipelines.insert(kernel_id.clone(), pipeline.clone());

        Ok(pipeline)
    }
}

impl ComputeServer for WgpuServer {
    type Kernel = Box<dyn CubeTask<AutoCompiler>>;
    type Storage = WgpuStorage;
    type Info = wgpu::Backend;

    fn logger(&self) -> Arc<ServerLogger> {
        self.scheduler.logger.clone()
    }

    fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }

    fn staging(&mut self, _sizes: &[usize], _stream_id: StreamId) -> Result<Vec<Bytes>, IoError> {
        // TODO: Check if using a staging buffer is useful here.
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        })
    }

    fn create(
        &mut self,
        descriptors: Vec<AllocationDescriptor<'_>>,
        stream_id: StreamId,
    ) -> Result<Vec<Allocation>, IoError> {
        let align = self.device.limits().min_storage_buffer_offset_alignment as usize;
        let strides = descriptors
            .iter()
            .map(|desc| contiguous_strides(desc.shape))
            .collect::<Vec<_>>();
        let sizes = descriptors
            .iter()
            .map(|desc| desc.shape.iter().product::<usize>() * desc.elem_size)
            .collect::<Vec<_>>();
        let total_size = sizes
            .iter()
            .map(|it| it.next_multiple_of(align))
            .sum::<usize>();

        let stream = self.scheduler.stream(&stream_id);
        let mem_handle = stream.empty(total_size as u64, stream_id)?;
        let handles = offset_handles(mem_handle, &sizes, align);

        Ok(handles
            .into_iter()
            .zip(strides)
            .map(|(handle, strides)| Allocation::new(handle, strides))
            .collect())
    }

    fn read<'a>(
        &mut self,
        descriptors: Vec<CopyDescriptor<'a>>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, IoError>> {
        let mut streams = vec![stream_id];
        let mut resources = Vec::with_capacity(descriptors.len());
        for desc in descriptors {
            if contiguous_strides(desc.shape) != desc.strides {
                return Box::pin(async {
                    Err(IoError::UnsupportedStrides {
                        backtrace: BackTrace::capture(),
                    })
                });
            }
            if !streams.contains(&desc.binding.stream) {
                streams.push(desc.binding.stream);
            }
            let stream = self.scheduler.stream(&desc.binding.stream);
            let resource = match stream.mem_manage.get_resource(desc.binding) {
                Ok(val) => val,
                Err(err) => return Box::pin(async move { Err(err) }),
            };
            resources.push((resource, desc.shape.to_vec(), desc.elem_size));
        }

        self.scheduler.execute_streams(streams);
        let stream = self.scheduler.stream(&stream_id);
        stream.read_resources(resources)
    }

    fn write(
        &mut self,
        descriptors: Vec<(CopyDescriptor<'_>, Bytes)>,
        stream_id: StreamId,
    ) -> Result<(), IoError> {
        for (desc, data) in descriptors {
            if contiguous_strides(desc.shape) != desc.strides {
                return Err(IoError::UnsupportedStrides {
                    backtrace: BackTrace::capture(),
                });
            }

            let stream = self.scheduler.stream(&desc.binding.stream);
            let resource = stream.mem_manage.get_resource(desc.binding.clone())?;
            let task = ScheduleTask::Write {
                data,
                buffer: resource,
            };

            self.scheduler.register(stream_id, task, [].into_iter());
        }

        Ok(())
    }

    fn get_resource(
        &mut self,
        binding: Binding,
        stream_id: StreamId,
    ) -> BindingResource<WgpuResource> {
        let mut streams = vec![stream_id];
        if binding.stream != stream_id {
            streams.push(binding.stream);
        }
        self.scheduler.execute_streams(streams);
        let stream = self.scheduler.stream(&binding.stream);
        let resource = stream.mem_manage.get_resource(binding.clone()).unwrap();
        BindingResource::new(binding, resource)
    }

    unsafe fn launch(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: Bindings,
        mode: ExecutionMode,
        stream_id: StreamId,
    ) -> Result<(), LaunchError> {
        let pipeline = self.pipeline(kernel, mode)?;
        let buffers = bindings.buffers.clone();
        let resources = self.prepare_bindings(bindings);
        let task = ScheduleTask::Execute {
            pipeline,
            count,
            resources,
        };

        self.scheduler.register(stream_id, task, buffers.iter());

        Ok(())
    }

    fn flush(&mut self, stream_id: StreamId) {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.flush()
    }

    /// Returns the total time of GPU work this sync completes.
    fn sync(&mut self, stream_id: StreamId) -> DynFut<Result<(), ExecutionError>> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.sync()
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

    fn memory_usage(
        &mut self,
        stream_id: StreamId,
    ) -> cubecl_runtime::memory_management::MemoryUsage {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.mem_manage.memory_usage()
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

pub(crate) fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let rank = shape.len();
    let mut strides = vec![1; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
