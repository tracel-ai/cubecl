use crate::{
    config::streaming::StreamingLogLevel,
    logging::ServerLogger,
    server::HandleBinding,
    stream::{StreamFactory, StreamPool},
};
use alloc::{format, sync::Arc, vec, vec::Vec};
use cubecl_common::stream_id::StreamId;

/// Defines a trait for a scheduler stream backend, specifying the types and behavior for task scheduling.
pub trait SchedulerStreamBackend {
    /// Type representing a task.
    type Task: core::fmt::Debug;
    /// Type representing a stream.
    type Stream: core::fmt::Debug;
    /// Type for the stream factory, which creates streams of type `Self::Stream`.
    type Factory: StreamFactory<Stream = Self::Stream>;

    /// Enqueues a task onto a given stream for execution.
    fn enqueue(task: Self::Task, stream: &mut Self::Stream);
    /// Flush the inner stream queue to ensure ordering between different streams.
    fn flush(stream: &mut Self::Stream);
    /// Returns a mutable reference to the stream factory.
    fn factory(&mut self) -> &mut Self::Factory;
}

/// Represents a multi-stream scheduler that manages task execution across multiple streams.
#[derive(Debug)]
pub struct SchedulerMultiStream<B: SchedulerStreamBackend> {
    /// Pool of streams managed by the scheduler.
    pool: StreamPool<SchedulerPoolMarker<B>>,
    /// Strategy for scheduling tasks (e.g., Interleave or Sequential).
    strategy: SchedulerStrategy,
    /// Maximum number of tasks allowed per stream before execution is triggered.
    max_tasks: usize,
    /// Server logger.
    pub logger: Arc<ServerLogger>,
}

/// Defines the scheduling strategy for task execution.
#[derive(Debug)]
pub enum SchedulerStrategy {
    /// Tasks from different streams are interleaved during execution.
    Interleave,
    /// Tasks from each stream are executed sequentially.
    Sequential,
}

/// Represents a single stream that holds tasks and a backend stream.
#[derive(Debug)]
pub struct Stream<B: SchedulerStreamBackend> {
    /// List of tasks queued for execution in this stream.
    tasks: Vec<B::Task>,
    /// The backend stream used for task execution.
    stream: B::Stream,
}

impl<B: SchedulerStreamBackend> Stream<B> {
    /// Flushes all tasks from the stream, returning them and clearing the internal task list.
    fn flush(&mut self) -> Vec<B::Task> {
        let mut returned = Vec::with_capacity(self.tasks.capacity());
        core::mem::swap(&mut returned, &mut self.tasks);
        returned
    }
}

#[derive(Debug)]
struct SchedulerPoolMarker<B: SchedulerStreamBackend> {
    backend: B,
}

impl<B: SchedulerStreamBackend> StreamFactory for SchedulerPoolMarker<B> {
    // The type of stream produced by this factory.
    type Stream = Stream<B>;

    // Creates a new stream with an empty task list and a backend stream.
    fn create(&mut self) -> Self::Stream {
        Stream {
            tasks: Vec::new(),
            // Uses the backend's factory to create a new stream.
            stream: self.backend.factory().create(),
        }
    }
}

/// Options for configuring a `SchedulerMultiStream`.
#[derive(Debug)]
pub struct SchedulerMultiStreamOptions {
    /// Maximum number of streams allowed in the pool.
    pub max_streams: u8,
    /// Maximum number of tasks per stream before execution is triggered.
    pub max_tasks: usize,
    /// The scheduling strategy to use.
    pub strategy: SchedulerStrategy,
}

impl<B: SchedulerStreamBackend> SchedulerMultiStream<B> {
    /// Creates a new `SchedulerMultiStream` with the given backend and options.
    pub fn new(
        logger: Arc<ServerLogger>,
        backend: B,
        options: SchedulerMultiStreamOptions,
    ) -> Self {
        Self {
            pool: StreamPool::new(SchedulerPoolMarker { backend }, options.max_streams, 0),
            max_tasks: options.max_tasks,
            strategy: options.strategy,
            logger,
        }
    }

    /// Returns a mutable reference to the backend stream for a given stream ID.
    pub fn stream(&mut self, stream_id: &StreamId) -> &mut B::Stream {
        let stream = self.pool.get_mut(stream_id);
        &mut stream.stream
    }

    /// Registers a task for execution on a specific stream, ensuring stream alignment.
    pub fn register<'a>(
        &mut self,
        stream_id: StreamId,
        task: B::Task,
        bindings: impl Iterator<Item = &'a HandleBinding>,
    ) {
        // Align streams to ensure dependencies are handled correctly.
        self.align_streams(stream_id, bindings);

        // Get the stream for the given stream ID and add the task to its queue.
        let current = self.pool.get_mut(&stream_id);
        current.tasks.push(task);

        // If the task queue exceeds the maximum, execute the stream.
        if current.tasks.len() >= self.max_tasks {
            self.execute_streams(vec![stream_id]);
        }
    }

    /// Aligns streams by flushing tasks from streams that conflict with the given bindings.
    pub(crate) fn align_streams<'a>(
        &mut self,
        stream_id: StreamId,
        bindings: impl Iterator<Item = &'a HandleBinding>,
    ) {
        let mut to_flush = Vec::new();
        // Get the index of the target stream.
        let index = self.pool.stream_index(&stream_id);

        // Identify streams that need to be flushed due to conflicting bindings.
        for binding in bindings {
            let index_stream = self.pool.stream_index(&binding.stream);
            if index != index_stream {
                to_flush.push(binding.stream);

                self.logger.log_streaming(
                    |level| matches!(level, StreamingLogLevel::Full),
                    || format!("Binding on {} is shared on {}", binding.stream, stream_id),
                );
            }
        }

        // If no streams need flushing, return early.
        if to_flush.is_empty() {
            return;
        }

        self.logger.log_streaming(
            |level| !matches!(level, StreamingLogLevel::Disabled),
            || {
                format!(
                    "Flushing streams {to_flush:?} before registering more tasks on {stream_id}"
                )
            },
        );
        // Execute the streams that need to be flushed.
        self.execute_streams(to_flush);
    }

    /// Executes tasks from the specified streams based on the scheduling strategy.
    pub fn execute_streams(&mut self, stream_ids: Vec<StreamId>) {
        let mut indices = Vec::with_capacity(stream_ids.len());
        // Collect unique stream indices to avoid redundant processing.
        for id in stream_ids {
            let index = self.pool.stream_index(&id);
            if !indices.contains(&index) {
                indices.push(index);
            }
        }

        // Create schedules for each stream to be executed.
        let mut schedules = Vec::new();
        for index in indices {
            let stream = unsafe { self.pool.get_mut_index(index) }; // Note: `unsafe` usage assumes valid index.
            let tasks = stream.flush();
            let num_tasks = tasks.len();

            schedules.push(Schedule {
                tasks: tasks.into_iter(),
                num_tasks,
                stream_index: index,
            });
        }

        // If no schedules were created, return early.
        if schedules.is_empty() {
            return;
        }

        // Execute schedules based on the configured strategy.
        match self.strategy {
            SchedulerStrategy::Interleave => self.execute_schedules_interleave(schedules),
            SchedulerStrategy::Sequential => self.execute_schedules_sequence(schedules),
        }
    }

    /// Executes schedules sequentially, processing each stream's tasks in order.
    fn execute_schedules_sequence(&mut self, schedules: Vec<Schedule<B>>) {
        for schedule in schedules {
            let stream = unsafe { self.pool.get_mut_index(schedule.stream_index) }; // Note: `unsafe` usage assumes valid index.
            for task in schedule.tasks {
                // Enqueue each task on the stream.
                B::enqueue(task, &mut stream.stream);
            }

            // Makes sure the tasks are ordered on the compute queue.
            B::flush(&mut stream.stream);
        }
    }

    //// Executes schedules in an interleaved manner, alternating tasks from different streams.
    ///
    /// We chose the first stream as the one executing the tasks, ensuring proper ordering by
    /// flushing all other streams first and flushing the execution stream at the end.
    /// This way, we ensure that most tasks are actually interleaved on the real compute queue
    /// shared across all streams.
    fn execute_schedules_interleave(&mut self, mut schedules: Vec<Schedule<B>>) {
        // Makes sure the tasks are ordered on the compute queue.
        for schedule in schedules.iter_mut().skip(1) {
            let stream = unsafe { self.pool.get_mut_index(schedule.stream_index) };
            B::flush(&mut stream.stream);
        }

        let execution_index = schedules.first().expect("At least one stream").stream_index;
        let stream = unsafe { self.pool.get_mut_index(execution_index) };

        // Find the maximum number of tasks across all schedules.
        let num_tasks_max = schedules
            .iter()
            .map(|s| s.num_tasks)
            .max()
            .expect("At least one schedule");

        // Iterate through tasks, interleaving them across streams.
        for _ in 0..num_tasks_max {
            for schedule in schedules.iter_mut() {
                // If there are tasks remaining in the schedule, enqueue the next one.
                if let Some(task) = schedule.tasks.next() {
                    B::enqueue(task, &mut stream.stream);
                }
            }
        }

        // Making sure all tasks are registered to the queue.
        B::flush(&mut stream.stream);
    }
}

// Represents a schedule for executing tasks on a specific stream.
struct Schedule<B: SchedulerStreamBackend> {
    // Iterator over the tasks to be executed.
    tasks: alloc::vec::IntoIter<B::Task>,
    // Number of tasks in the schedule.
    num_tasks: usize,
    // Index of the stream in the pool.
    stream_index: usize,
}
