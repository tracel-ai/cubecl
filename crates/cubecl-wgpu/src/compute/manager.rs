use crate::compiler::base::WgpuCompiler;

use super::{
    stream::{PipelineDispatch, WgpuStream},
    timestamps::KernelTimestamps,
    WgpuServer,
};
use cubecl_common::stream::StreamId;
use cubecl_runtime::{storage::BindingResource, TimestampsResult};
use std::{
    collections::{BTreeMap, BTreeSet},
    future::Future,
    pin::Pin,
    sync::Arc,
};

#[derive(Debug)]
pub struct WgpuStreamManager {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    timestamps: KernelTimestamps,
    tasks_max: usize,
    streams: BTreeMap<StreamId, WgpuStream>,
    buffer2stream: BTreeMap<wgpu::Id<wgpu::Buffer>, StreamId>,
    stream2buffer: BTreeMap<StreamId, BTreeSet<wgpu::Id<wgpu::Buffer>>>,
}

impl WgpuStreamManager {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        timestamps: KernelTimestamps,
        tasks_max: usize,
    ) -> Self {
        Self {
            device,
            queue,
            timestamps,
            tasks_max,
            streams: BTreeMap::new(),
            buffer2stream: BTreeMap::new(),
            stream2buffer: BTreeMap::new(),
        }
    }

    pub fn register<C: WgpuCompiler>(
        &mut self,
        pipeline: Arc<wgpu::ComputePipeline>,
        resources: Vec<BindingResource<WgpuServer<C>>>,
        dispatch: PipelineDispatch<C>,
    ) -> bool {
        let stream_id = StreamId::current();
        self.update_timelines(&resources, stream_id);

        let stream = match self.streams.get_mut(&stream_id) {
            Some(stream) => stream,
            None => {
                self.streams.insert(
                    stream_id,
                    WgpuStream::new(
                        self.device.clone(),
                        self.queue.clone(),
                        self.timestamps.duplicate(&self.device),
                        self.tasks_max,
                    ),
                );
                self.streams.get_mut(&stream_id).unwrap()
            }
        };

        stream.register(pipeline, resources, dispatch)
    }

    pub fn read_buffer(
        &mut self,
        buffer: &wgpu::Buffer,
        offset: u64,
        size: u64,
    ) -> impl Future<Output = Vec<u8>> + 'static {
        let stream_id = StreamId::current();

        if let Some(stream_id) = self.buffer2stream.get(&buffer.global_id()) {
            let stream = self.streams.get_mut(stream_id).unwrap();
            stream.flush();
            let fut = stream.read_buffer(buffer, offset, size);
            self.on_stream_flush(*stream_id);
            return fut;
        }

        match self.streams.get_mut(&stream_id) {
            Some(stream) => stream.read_buffer(buffer, offset, size),
            None => panic!(""),
        }
    }

    pub fn sync(&mut self) -> Pin<Box<dyn Future<Output = ()> + Send + 'static>> {
        let stream_id = StreamId::current();

        match self.streams.get_mut(&stream_id) {
            Some(stream) => {
                let fut = stream.sync();
                self.on_stream_flush(stream_id);
                fut
            }
            None => Box::pin(async move {}),
        }
    }

    pub fn sync_elapsed(
        &mut self,
    ) -> Pin<Box<dyn Future<Output = TimestampsResult> + Send + 'static>> {
        let stream_id = StreamId::current();

        match self.streams.get_mut(&stream_id) {
            Some(stream) => {
                let fut = stream.sync_elapsed();
                self.on_stream_flush(stream_id);
                fut
            }
            None => Box::pin(async move {
                TimestampsResult::Err(cubecl_runtime::TimestampsError::Unavailable)
            }),
        }
    }

    pub fn flush(&mut self) {
        let stream_id = StreamId::current();

        match self.streams.get_mut(&stream_id) {
            Some(stream) => {
                stream.flush();
                self.on_stream_flush(stream_id);
            }
            None => {}
        }
    }

    pub fn enable_timestamps(&mut self) {
        self.timestamps.enable(&self.device);
        for stream in self.streams.values_mut() {
            stream.timestamps.enable(&self.device);
        }
    }

    pub fn disable_timestamps(&mut self) {
        self.timestamps.disable();
        for stream in self.streams.values_mut() {
            stream.timestamps.disable();
        }
    }

    fn on_stream_flush(&mut self, stream_id: StreamId) {
        if let Some(bufs) = self.stream2buffer.remove(&stream_id) {
            for buf in bufs {
                self.buffer2stream.remove(&buf);
            }
        }
    }

    fn update_timelines<C: WgpuCompiler>(
        &mut self,
        resources: &[BindingResource<WgpuServer<C>>],
        stream_id: StreamId,
    ) {
        return;
        let mut flushes = Vec::new();
        let buffers = match self.stream2buffer.get_mut(&stream_id) {
            Some(val) => val,
            None => {
                self.stream2buffer.insert(stream_id, Default::default());
                self.stream2buffer.get_mut(&stream_id).unwrap()
            }
        };
        resources.iter().for_each(|r| {
            let id = r.resource().buffer.global_id();
            if let Some(id) = self.buffer2stream.get(&id) {
                flushes.push(*id);
            } else {
                self.buffer2stream.insert(id, stream_id);
            }
            buffers.insert(id);
        });

        for id in flushes {
            if let Some(stream) = self.streams.get_mut(&id) {
                stream.flush();
            }
            self.on_stream_flush(id);
        }
    }
}
