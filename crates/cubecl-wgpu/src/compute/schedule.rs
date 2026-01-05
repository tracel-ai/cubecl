use crate::{WgpuResource, stream::WgpuStream};
use alloc::sync::Arc;
use cubecl_common::{bytes::Bytes, profile::TimingMethod};
use cubecl_core::{
    CubeCount, MemoryConfiguration,
    ir::StorageType,
    server::{MetadataBinding, ScalarBinding},
};
use cubecl_ir::MemoryDeviceProperties;
use cubecl_runtime::{
    logging::ServerLogger,
    stream::{StreamFactory, scheduler::SchedulerStreamBackend},
};
use std::collections::BTreeMap;

/// Defines tasks that can be scheduled on a WGPU stream.
#[derive(Debug)]
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

/// Represents a collection of resources and bindings for a compute task.
#[derive(Debug)]
pub struct BindingsResource {
    /// List of WGPU resources used in the task.
    pub resources: Vec<WgpuResource>,
    /// Metadata for uniform bindings.
    pub metadata: MetadataBinding,
    /// Scalar values mapped by their storage type.
    pub scalars: BTreeMap<StorageType, ScalarBinding>,
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
}

impl StreamFactory for WgpuStreamFactory {
    type Stream = WgpuStream;

    fn create(&mut self) -> Self::Stream {
        WgpuStream::new(
            self.device.clone(),
            self.queue.clone(),
            self.memory_properties.clone(),
            self.memory_config.clone(),
            self.timing_method,
            self.tasks_max,
            self.logger.clone(),
        )
    }
}

impl ScheduledWgpuBackend {
    /// Creates a new `ScheduledWgpuBackend` with the given WGPU device, queue, and configurations.
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        timing_method: TimingMethod,
        tasks_max: usize,
        logger: Arc<ServerLogger>,
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
            },
        }
    }
}

impl BindingsResource {
    /// Converts metadata and scalar bindings into WGPU resources for a stream.
    pub fn into_resources(mut self, stream: &mut WgpuStream) -> Vec<WgpuResource> {
        // If metadata contains data, create a uniform buffer for it.
        if !self.metadata.data.is_empty() {
            let info = stream.create_uniform(bytemuck::cast_slice(&self.metadata.data));
            self.resources.push(info);
        }

        // Convert scalar bindings into uniform buffers and add them to the resources.
        self.resources.extend(
            self.scalars
                .values()
                .map(|s| stream.create_uniform(s.data())),
        );

        // Return the complete list of resources.
        self.resources
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
        stream.flush();
    }

    fn factory(&mut self) -> &mut Self::Factory {
        &mut self.factory
    }
}
