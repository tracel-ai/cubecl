use async_channel::Sender;
use cubecl_common::stream::StreamId;
use std::{
    future::Future,
    pin::Pin,
    sync::{atomic::AtomicBool, Arc},
};

use crate::compiler::base::WgpuCompiler;

use super::{
    processor::{AsyncTask, Message, PipelineDispatch, SyncTask, Task, WgpuProcessor},
    timestamps::KernelTimestamps,
    WgpuServer,
};
use cubecl_runtime::{storage::BindingResource, TimestampsResult};
use wgpu::ComputePipeline;

#[derive(Debug)]
pub struct WgpuS<C: WgpuCompiler> {
    caller: Sender<Message<C>>,
    should_flush: Arc<AtomicBool>,
}

impl<C: WgpuCompiler> WgpuS<C> {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        timestamps: KernelTimestamps,
        tasks_max: usize,
    ) -> Self {
        let processor = WgpuProcessor::new(device.clone(), queue.clone(), timestamps, tasks_max);
        let (caller, should_flush) = processor.start();

        Self {
            caller,
            should_flush,
        }
    }

    pub fn register(
        &mut self,
        pipeline: Arc<ComputePipeline>,
        resources: Vec<BindingResource<WgpuServer<C>>>,
        dispatch: PipelineDispatch<C>,
    ) -> bool {
        let msg = Message::new(
            StreamId::current(),
            Task::Async(AsyncTask {
                pipeline,
                resources,
                dispatch,
            }),
        );

        self.caller.send_blocking(msg).unwrap();
        self.should_flush
            .swap(false, std::sync::atomic::Ordering::Relaxed)
    }

    pub fn read_buffer(
        &mut self,
        buffer: Arc<wgpu::Buffer>,
        offset: u64,
        size: u64,
    ) -> impl Future<Output = Vec<u8>> + 'static {
        let (sender, rev) = async_channel::bounded(10);
        let msg = Message::new(
            StreamId::current(),
            Task::Sync(SyncTask::Read {
                buffer,
                offset,
                size,
                callback: sender,
            }),
        );

        self.caller.send_blocking(msg).unwrap();

        async move {
            match rev.recv().await {
                Ok(data) => data,
                Err(err) => panic!("{err:?}"),
            }
        }
    }

    pub fn sync_elapsed(
        &mut self,
    ) -> Pin<Box<dyn Future<Output = TimestampsResult> + Send + 'static>> {
        let (sender, rev) = async_channel::bounded(10);
        let msg = Message::new(
            StreamId::current(),
            Task::Sync(SyncTask::SyncElapsed { callback: sender }),
        );

        self.caller.send_blocking(msg).unwrap();

        Box::pin(async move {
            match rev.recv().await {
                Ok(data) => data,
                Err(err) => panic!("{err:?}"),
            }
        })
    }

    pub fn sync(&mut self) -> Pin<Box<dyn Future<Output = ()> + Send + 'static>> {
        let (sender, rev) = async_channel::bounded(10);
        let msg = Message::new(
            StreamId::current(),
            Task::Sync(SyncTask::Sync { callback: sender }),
        );

        self.caller.send_blocking(msg).unwrap();

        Box::pin(async move {
            match rev.recv().await {
                Ok(data) => data,
                Err(err) => panic!("{err:?}"),
            }
        })
    }

    pub fn enable_timestamps(&mut self) {
        let msg = Message::new(StreamId::current(), Task::EnableTimestamp);
        self.caller.send_blocking(msg).unwrap();
    }

    pub fn disable_timestamps(&mut self) {
        let msg = Message::new(StreamId::current(), Task::DisableTimestamp);
        self.caller.send_blocking(msg).unwrap();
    }

    pub fn flush(&mut self) {
        cubecl_common::future::block_on(self.sync());
    }
}
