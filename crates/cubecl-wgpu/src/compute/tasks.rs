use std::collections::BTreeMap;

use crate::{WgpuResource, stream::WgpuStream};
use alloc::sync::Arc;
use cubecl_common::profile::TimingMethod;
use cubecl_core::{
    CubeCount, MemoryConfiguration,
    ir::StorageType,
    server::{MetadataBinding, ScalarBinding},
};
use cubecl_runtime::{
    memory_management::MemoryDeviceProperties,
    stream::{StreamFactory, scheduler::SchedulerStreamBackend},
};

#[derive(Debug)]
pub(crate) enum LazyTask {
    Write {
        data: Vec<u8>,
        buffer: WgpuResource,
    },
    Execute {
        pipeline: Arc<wgpu::ComputePipeline>,
        count: CubeCount,
        resources: BindingsResource,
    },
}

#[derive(Debug)]
pub(crate) struct ScheduledWgpuBackend {
    factory: WgpuStreamFactory,
}

#[derive(Debug)]
pub struct WgpuStreamFactory {
    device: wgpu::Device,
    queue: wgpu::Queue,
    memory_properties: MemoryDeviceProperties,
    memory_config: MemoryConfiguration,
    timing_method: TimingMethod,
    tasks_max: usize,
}

impl StreamFactory for WgpuStreamFactory {
    type Stream = WgpuStream;

    fn create(&mut self) -> Self::Stream {
        WgpuStream::new(
            self.device.clone(),
            self.queue.clone(),
            self.memory_properties.clone(),
            self.memory_config.clone(),
            self.timing_method.clone(),
            self.tasks_max,
        )
    }
}

impl ScheduledWgpuBackend {
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        timing_method: TimingMethod,
        tasks_max: usize,
    ) -> Self {
        let factory = WgpuStreamFactory {
            device,
            queue,
            memory_properties,
            memory_config,
            timing_method,
            tasks_max,
        };

        Self { factory }
    }
}

#[derive(Debug)]
pub struct BindingsResource {
    pub resources: Vec<WgpuResource>,
    pub metadata: MetadataBinding,
    pub scalars: BTreeMap<StorageType, ScalarBinding>,
}

impl BindingsResource {
    pub fn into_resources(mut self, stream: &mut WgpuStream) -> Vec<WgpuResource> {
        if !self.metadata.data.is_empty() {
            let info = stream.create_uniform(bytemuck::cast_slice(&self.metadata.data));
            self.resources.push(info);
        }

        self.resources.extend(
            self.scalars
                .values()
                .map(|s| stream.create_uniform(s.data())),
        );

        self.resources
    }
}

impl SchedulerStreamBackend for ScheduledWgpuBackend {
    type Task = LazyTask;
    type Stream = WgpuStream;
    type Factory = WgpuStreamFactory;

    fn enqueue(task: Self::Task, stream: &mut Self::Stream) {
        stream.execute_task(task);
    }

    fn factory(&mut self) -> &mut Self::Factory {
        &mut self.factory
    }
}
