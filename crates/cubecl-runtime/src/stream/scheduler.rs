use crate::{
    server::Binding,
    stream::{StreamFactory, StreamPool},
};
use cubecl_common::stream_id::StreamId;

pub trait SchedulerStreamBackend {
    type Task: SchedulerTask;
    type Stream: core::fmt::Debug;
    type Factory: StreamFactory<Stream = Self::Stream>;

    fn enqueue(task: Self::Task, stream: &mut Self::Stream);
    fn factory(&mut self) -> &mut Self::Factory;
}

pub trait SchedulerTask: core::fmt::Debug {}

#[derive(Debug)]
pub struct SchedulerMultiStream<B: SchedulerStreamBackend> {
    pool: StreamPool<SchedulerPoolMarker<B>>,
}

#[derive(Debug)]
pub struct Stream<B: SchedulerStreamBackend> {
    tasks: Vec<B::Task>,
    stream: B::Stream,
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
    backend: B,
}

impl<B: SchedulerStreamBackend> StreamFactory for SchedulerPoolMarker<B> {
    type Stream = Stream<B>;

    fn create(&mut self) -> Self::Stream {
        Stream {
            tasks: Vec::new(),
            stream: self.backend.factory().create(),
        }
    }
}

impl<B: SchedulerStreamBackend> SchedulerMultiStream<B> {
    pub fn new(backend: B, max_streams: u8) -> Self {
        Self {
            pool: StreamPool::new(SchedulerPoolMarker { backend }, max_streams, 0),
        }
    }

    pub fn stream(&mut self, stream_id: &StreamId) -> &mut B::Stream {
        let stream = self.pool.get_mut(stream_id);
        &mut stream.stream
    }

    pub fn backend(&mut self) -> &mut B {
        &mut self.pool.factory.backend
    }

    pub fn register<'a>(
        &mut self,
        stream_id: StreamId,
        task: B::Task,
        bindings: impl Iterator<Item = &'a Binding>,
    ) {
        self.align_streams(stream_id, bindings);

        let current = self.pool.get_mut(&stream_id);
        current.tasks.push(task);

        if current.tasks.len() > 32 {
            self.execute_streams(vec![stream_id]);
        }
    }

    pub(crate) fn align_streams<'a>(
        &mut self,
        stream_id: StreamId,
        bindings: impl Iterator<Item = &'a Binding>,
    ) {
        let mut to_flush = Vec::new();
        let index = self.pool.stream_index(&stream_id);

        for binding in bindings {
            let index_stream = self.pool.stream_index(&binding.stream);
            if index != index_stream {
                to_flush.push(binding.stream);
            }
        }

        if to_flush.is_empty() {
            return;
        }

        log::info!("Flush...");
        self.execute_streams(to_flush);
    }

    pub fn execute_streams(&mut self, stream_ids: Vec<StreamId>) {
        let mut indices = Vec::with_capacity(stream_ids.len());
        for id in stream_ids {
            let index = self.pool.stream_index(&id);
            if !indices.contains(&index) {
                indices.push(index);
            }
        }

        let mut metadata = Vec::new();

        for index in indices {
            let stream = unsafe { self.pool.get_mut_index(index) };
            let tasks = stream.flush();
            let num = tasks.len();
            metadata.push((index, num, tasks.into_iter()));
        }

        if metadata.is_empty() {
            return;
        }
        // println!("{metadata:?}");

        let total: usize = metadata.iter().map(|i| i.1).sum();
        log::info!("Execute scheduled {total} tasks ...");

        let num_flushes = metadata.len();

        let mut finished = vec![false; num_flushes];
        let mut num_finished = 0;

        loop {
            if num_finished == num_flushes {
                break;
            }

            for i in 0..num_flushes {
                let meta = &mut metadata[i];
                let index = meta.0;
                if let Some(task) = meta.2.next() {
                    let stream = unsafe { self.pool.get_mut_index(index) };
                    B::enqueue(task, &mut stream.stream);
                } else if !finished[i] {
                    finished[i] = true;
                    num_finished += 1;
                }
            }
        }
    }
}
