use crate::{CompilerInfo, ParamsTransfer, WgpuResource, stream::WgpuStream};
use alloc::sync::Arc;
use cubecl_common::{bytes::Bytes, profile::TimingMethod};
use cubecl_core::{
    CubeCount, MemoryConfiguration,
    server::{MetadataBindingInfo, StreamErrorMode},
    zspace::SmallVec,
};
use cubecl_ir::MemoryDeviceProperties;
use cubecl_runtime::{
    logging::ServerLogger,
    stream::{StreamFactory, scheduler::SchedulerStreamBackend},
};

/// Defines tasks that can be scheduled on a WGPU stream.
pub enum ScheduleTask {
    /// Represents a task to write data to a buffer.
    Write {
        /// The data to be written.
        data: Bytes,
        /// The target buffer resource.
        buffer: WgpuResource,
    },
    /// Represents a task to execute a compute pipeline.
    Execute {
        /// The compute pipeline to execute.
        pipeline: Arc<wgpu::ComputePipeline>,
        /// The number of workgroups to dispatch.
        count: CubeCount,
        /// The resources (bindings) required for execution.
        resources: BindingsResource,
    },
}

impl core::fmt::Debug for ScheduleTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Write { data, .. } => f.write_fmt(format_args!("Write(bytes={})", data.len())),
            Self::Execute {
                count, resources, ..
            } => f.write_fmt(format_args!(
                "Execute(resources={}, cube_count={count:?})",
                resources.resources.len()
            )),
        }
    }
}

/// Represents a collection of resources and bindings for a compute task.
#[derive(Debug)]
pub struct BindingsResource {
    /// List of WGPU resources used in the task.
    pub resources: Vec<WgpuResource>,
    /// Metadata for uniform bindings.
    pub info: MetadataBindingInfo,
    /// Which compiler was used. This determines the passing strategy of params.
    /// WGSL and metal use bindings, Vulkan uses buffer addresses sent via a uniform buffer.
    pub compiler_info: CompilerInfo,
}

/// Represents a WGPU backend for scheduling tasks on streams.
#[derive(Debug)]
pub struct ScheduledWgpuBackend {
    /// Factory for creating WGPU streams.
    factory: WgpuStreamFactory,
}

/// Factory for creating WGPU streams with specific configurations.
#[derive(Debug)]
pub struct WgpuStreamFactory {
    device: wgpu::Device,
    queue: wgpu::Queue,
    memory_properties: MemoryDeviceProperties,
    memory_config: MemoryConfiguration,
    timing_method: TimingMethod,
    tasks_max: usize,
    logger: Arc<ServerLogger>,
    count: u64,
    use_vulkan_compiler: bool,
}

impl StreamFactory for WgpuStreamFactory {
    type Stream = WgpuStream;

    fn create(&mut self) -> Self::Stream {
        self.count += 1;

        WgpuStream::new(
            self.device.clone(),
            self.queue.clone(),
            self.memory_properties.clone(),
            self.memory_config.clone(),
            self.timing_method,
            self.tasks_max,
            self.logger.clone(),
            self.use_vulkan_compiler,
        )
    }
}

impl ScheduledWgpuBackend {
    /// Creates a new `ScheduledWgpuBackend` with the given WGPU device, queue, and configurations.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        timing_method: TimingMethod,
        tasks_max: usize,
        logger: Arc<ServerLogger>,
        use_vulkan_compiler: bool,
    ) -> Self {
        Self {
            factory: WgpuStreamFactory {
                device,
                queue,
                memory_properties,
                memory_config,
                timing_method,
                tasks_max,
                logger,
                count: 0,
                use_vulkan_compiler,
            },
        }
    }
}

pub type Addresses = SmallVec<[u64; 8]>;

impl BindingsResource {
    /// Converts metadata and scalar bindings into WGPU resources for a stream.
    pub fn into_resources(
        mut self,
        stream: &mut WgpuStream,
    ) -> (Vec<WgpuResource>, Vec<WgpuResource>, Option<Addresses>) {
        let info = (!self.info.data.is_empty())
            .then(|| stream.create_uniform(bytemuck::cast_slice(&self.info.data)));
        match self.compiler_info {
            CompilerInfo::Vulkan { params_transfer } => {
                let addresses = self
                    .resources
                    .iter()
                    .chain(info.iter())
                    .map(|it| it.address.unwrap().get() + it.offset)
                    .collect::<Addresses>();
                if let Some(info) = info {
                    self.resources.push(info);
                }
                match params_transfer {
                    ParamsTransfer::Immediate => (vec![], self.resources, Some(addresses)),
                    ParamsTransfer::Uniform => {
                        let address_buffer =
                            stream.create_uniform(bytemuck::cast_slice(&addresses));
                        (vec![address_buffer], self.resources, None)
                    }
                }
            }
            _ => {
                if let Some(info) = info {
                    self.resources.push(info);
                }
                (self.resources, vec![], None)
            }
        }
    }
}

impl SchedulerStreamBackend for ScheduledWgpuBackend {
    type Task = ScheduleTask;
    type Stream = WgpuStream;
    type Factory = WgpuStreamFactory;

    fn enqueue(task: Self::Task, stream: &mut Self::Stream) {
        stream.enqueue_task(task);
    }

    fn flush(stream: &mut Self::Stream) {
        let _ = stream
            .flush(StreamErrorMode {
                ignore: true,
                flush: false,
            })
            .ok();
    }

    fn factory(&mut self) -> &mut Self::Factory {
        &mut self.factory
    }
}
