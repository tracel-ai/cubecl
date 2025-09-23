use std::collections::BTreeMap;

use crate::{WgpuResource, stream::WgpuStream};
use alloc::sync::Arc;
use cubecl_common::{profile::TimingMethod, stream_id::StreamId};
use cubecl_core::{
    CubeCount, MemoryConfiguration,
    ir::StorageType,
    server::{Bindings, MetadataBinding, ScalarBinding},
};
use cubecl_runtime::{
    memory_management::MemoryDeviceProperties,
    stream::{
        StreamFactory, StreamPool,
        scheduler::{SchedulerStreamBackend, SchedulerTask},
    },
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

impl SchedulerTask for LazyTask {}

#[derive(Debug)]
pub(crate) struct ScheduledWgpuBackend {
    pool: StreamPool<WgpuStreamFactory>,
}

#[derive(Debug)]
struct WgpuStreamFactory {
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

        Self {
            pool: StreamPool::new(factory, 1, 0),
        }
    }

    pub fn stream(&mut self, stream_id: &StreamId) -> &mut WgpuStream {
        self.pool.get_mut(stream_id)
    }

    pub fn bindings(&mut self, stream_id: &StreamId, bindings: Bindings) -> BindingsResource {
        // Store all the resources we'll be using. This could be eliminated if
        // there was a way to tie the lifetime of the resource to the memory handle.
        let resources = bindings
            .buffers
            .iter()
            .map(|b| {
                let stream = self.stream(&b.stream);
                stream.mem_manage.get_resource(b.clone())
            })
            .collect::<Vec<_>>();

        BindingsResource {
            resources,
            metadata: bindings.metadata,
            scalars: bindings.scalars,
        }
    }
}

#[derive(Debug)]
pub struct BindingsResource {
    resources: Vec<WgpuResource>,
    metadata: MetadataBinding,
    scalars: BTreeMap<StorageType, ScalarBinding>,
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

    fn execute(&mut self, tasks: impl Iterator<Item = (usize, Self::Task)>) {
        for (index, task) in tasks {
            let stream = unsafe { self.pool.get_mut_index(index) };
            stream.execute_task(task);
        }
    }
}
