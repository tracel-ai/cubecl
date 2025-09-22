use crate::{
    server::Binding,
    stream::{StreamFactory, StreamPool},
};
use core::marker::PhantomData;
use cubecl_common::stream_id::StreamId;

pub trait SchedulerStreamBackend {
    type Task: SchedulerTask;

    async fn execute(&mut self, tasks: impl Iterator<Item = Self::Task>);
}

pub trait SchedulerTask: core::fmt::Debug {
    fn stream_id(&self) -> StreamId;
    fn bindings<'a>(&'a self) -> Vec<&'a Binding>;
}

#[derive(Debug)]
pub struct SchedulerMultiStream<B: SchedulerStreamBackend> {
    pool: StreamPool<SchedulerPoolMarker<B>>,
    backend: B,
}

#[derive(Debug)]
pub struct Stream<B: SchedulerStreamBackend> {
    tasks: Vec<B::Task>,
}

impl<B: SchedulerStreamBackend> Stream<B> {
    fn flush(&mut self) -> Vec<B::Task> {
        let mut returned = Vec::new();
        core::mem::swap(&mut returned, &mut self.tasks);
        returned
    }
}

#[derive(Debug)]
pub struct SchedulerPoolMarker<B: SchedulerStreamBackend> {
    _p: PhantomData<B>,
}

impl<B: SchedulerStreamBackend> Default for SchedulerPoolMarker<B> {
    fn default() -> Self {
        Self {
            _p: Default::default(),
        }
    }
}

impl<B: SchedulerStreamBackend> StreamFactory for SchedulerPoolMarker<B> {
    type Stream = Stream<B>;

    fn create(&mut self) -> Self::Stream {
        Stream { tasks: Vec::new() }
    }
}

impl<B: SchedulerStreamBackend> SchedulerMultiStream<B> {
    pub fn new(backend: B, max_streams: u8) -> Self {
        Self {
            pool: StreamPool::new(SchedulerPoolMarker::default(), max_streams, 0),
            backend,
        }
    }

    pub async fn register(&mut self, task: B::Task) {
        let stream_id = task.stream_id();
        self.align_streams(stream_id, task.bindings()).await;

        let current = self.pool.get_mut(&stream_id);
        current.tasks.push(task);
    }

    pub(crate) async fn align_streams<'a>(
        &mut self,
        stream_id: StreamId,
        bindings: Vec<&'a Binding>,
    ) {
        let mut to_flush = Vec::new();

        for binding in bindings {
            if binding.stream != stream_id {
                to_flush.push(binding.stream);
            }
        }

        if to_flush.is_empty() {
            return;
        }

        self.execute_streams(to_flush).await;
    }

    pub(crate) async fn execute_streams(&mut self, stream_ids: Vec<StreamId>) {
        let mut metadata = Vec::new();

        for stream_id in stream_ids.into_iter() {
            let stream = self.pool.get_mut(&stream_id);
            let tasks = stream.flush();
            let num = tasks.len();
            metadata.push((stream_id, num, tasks.into_iter()));
        }

        if metadata.is_empty() {
            return;
        }

        let total: usize = metadata.iter().map(|i| i.1).sum();

        let num_flushes = metadata.len();
        let mut tasks = Vec::with_capacity(total);

        for i in 0..num_flushes {
            if let Some(task) = metadata[i].2.next() {
                tasks.push(task);
            }
        }

        self.backend.execute(tasks.into_iter()).await;
    }
}
