use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};

use cubecl_core::config::RuntimeConfig;
use cubecl_runtime::config::CubeClRuntimeConfig;

use crate::compute::{
    affinity::get_active_cores,
    threadpool::{
        ThreadTask, circular_buffer::CircularBuffer, compute_task::ComputeTask, scheduler::Worker,
    },
};

pub struct SimpleScheduler {
    threads_buffer: Vec<SimpleWorker>,
}

impl SimpleScheduler {
    pub fn new() -> Self {
        let config = CubeClRuntimeConfig::get();
        let max_streams = config.streaming.max_streams;

        let threads_buffer = get_active_cores()
            .map(|core_id| {
                let worker = SimpleWorker::new(max_streams as usize);
                worker.clone().spawn_thread(core_id);
                worker
            })
            .collect();

        Self { threads_buffer }
    }

    pub fn flush(&mut self, stream_index: usize) {
        for buffer in &mut self.threads_buffer {
            buffer.buffer.lock().to_flush.push_back(stream_index);
        }
    }

    pub fn send(&mut self, index: usize, task: ComputeTask) {
        self.threads_buffer[index].buffer.lock().push(task)
    }
}

/// Local thread buffer used for pushing task to a worker for the scheduler and for getting task for the worker
struct ThreadBuffer<T: ThreadTask> {
    /// Flush queue
    to_flush: VecDeque<usize>,
    /// Main stream storage
    streams: CircularBuffer<VecDeque<T>>,
    /// Empty streams available to be reused to avoid reallocation
    empty_streams: Vec<VecDeque<T>>,
    /// Association of stream id to position in `streams`
    streams_id: HashMap<usize, usize>,
}

impl<T: ThreadTask> ThreadBuffer<T> {
    /// Construct a ThreadBuffer with an empty reference to all thread_buffer
    fn new(capacity: usize) -> Self {
        let streams = CircularBuffer::new(capacity);
        let streams_id = HashMap::new();
        let empty_streams = Vec::new();
        let to_flush = VecDeque::new();
        Self {
            to_flush,
            streams,
            empty_streams,
            streams_id,
        }
    }

    /// This function assume that the fifo is not empty
    fn push_fifo(&mut self, fifo: VecDeque<T>) {
        let id = fifo.front().unwrap().get_stream_id();
        let back = self.streams.back();
        self.streams.push_back(fifo);
        self.streams_id.entry(id).insert_entry(back);
    }

    fn push_new(&mut self, elem: T) {
        let mut fifo = match self.empty_streams.pop() {
            Some(fifo) => fifo,
            None => VecDeque::new(),
        };
        fifo.push_back(elem);
        self.push_fifo(fifo);
    }

    /// Add new task to the local thread
    fn push(&mut self, elem: T) {
        match self.streams_id.get(&elem.get_stream_id()) {
            Some(&i) => {
                self.streams[i].push_back(elem);
            }
            None => self.push_new(elem),
        }
    }

    fn pop_local(&mut self) -> Option<T> {
        for _ in 0..self.streams.len() {
            if self.streams.is_empty() {
                return None;
            }
            let front = self.streams.front();
            let fifos = &mut self.streams[front];
            if fifos.is_empty() {
                let fifo = self.streams.pop_front().unwrap();
                self.empty_streams.push(fifo);
                continue;
            }
            if !fifos.front().unwrap().is_ready() {
                continue;
            }
            let elem = fifos.pop_front().unwrap();
            if fifos.is_empty() {
                let stream = self.streams.pop_front().unwrap();
                self.empty_streams.push(stream);
                self.streams_id.remove(&elem.get_stream_id());
            }
            return Some(elem);
        }
        return None;
    }

    /// Pop task for local execution
    fn pop(&mut self) -> Option<T> {
        if let Some(stream_id) = self.to_flush.front() {
            let elem = self.pop_id(*stream_id);
            if let None = elem {
                self.to_flush.pop_front();
            }
            return elem;
        }
        if let Some(elem) = self.pop_local() {
            return Some(elem);
        }
        None
    }

    /// Pop task with id for flushing
    fn pop_id(&mut self, thread_id: usize) -> Option<T> {
        let id = self.streams_id.get(&thread_id)?;
        self.empty_streams[*id].pop_front()
    }
}

#[derive(Clone)]
pub struct SimpleWorker {
    buffer: Arc<spin::Mutex<ThreadBuffer<ComputeTask>>>,
}

impl Worker for SimpleWorker {
    fn work(self) {
        loop {
            if let Some(compute_task) = self.buffer.lock().pop() {
                compute_task.compute();
            }

            std::hint::spin_loop();
        }
    }
}

impl SimpleWorker {
    pub fn new(capacity: usize) -> SimpleWorker {
        let buffer = Arc::new(spin::Mutex::new(ThreadBuffer::new(capacity)));
        SimpleWorker { buffer }
    }
}

#[cfg(test)]
mod tests {
    use super::{ThreadBuffer, ThreadTask};
    use std::cell::RefCell;

    thread_local! {
        static FAKE_TIME: RefCell<u64> = RefCell::new(0);
    }

    fn set_fake_time(time: u64) {
        FAKE_TIME.with(|t| {
            *t.borrow_mut() = time;
        });
    }

    fn get_fake_time() -> u64 {
        FAKE_TIME.with(|t| *t.borrow())
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct TestTask {
        stream_id: usize,
        value: usize,
        created_at: u64,
        duration: u64,
    }

    impl TestTask {
        fn new(stream_id: usize, value: usize, duration: u64) -> Self {
            Self {
                stream_id,
                value,
                created_at: get_fake_time(),
                duration,
            }
        }
    }

    impl ThreadTask for TestTask {
        fn get_stream_id(&self) -> usize {
            self.stream_id
        }

        fn is_ready(&self) -> bool {
            let elapsed = get_fake_time().saturating_sub(self.created_at);
            elapsed >= self.duration
        }
    }

    #[test]
    fn test_pop_empty_returns_none() {
        let mut buffer = ThreadBuffer::<TestTask>::new(8);
        assert_eq!(None, buffer.pop());
        assert_eq!(None, buffer.pop());
    }

    #[test]
    fn test_single_stream_fifo_pop_order() {
        set_fake_time(0);
        let mut buffer = ThreadBuffer::<TestTask>::new(8);

        buffer.push(TestTask::new(1, 10, 0));
        buffer.push(TestTask::new(1, 11, 0));
        buffer.push(TestTask::new(1, 12, 0));

        assert_eq!(Some(TestTask::new(1, 10, 0)), buffer.pop());
        assert_eq!(Some(TestTask::new(1, 11, 0)), buffer.pop());
        assert_eq!(Some(TestTask::new(1, 12, 0)), buffer.pop());
        assert_eq!(None, buffer.pop());
    }

    #[test]
    fn test_repush_same_stream_after_progress() {
        set_fake_time(0);
        let mut buffer = ThreadBuffer::<TestTask>::new(8);

        buffer.push(TestTask::new(7, 1, 0));
        buffer.push(TestTask::new(7, 2, 0));
        assert_eq!(Some(TestTask::new(7, 1, 0)), buffer.pop());

        buffer.push(TestTask::new(7, 3, 0));

        assert_eq!(Some(TestTask::new(7, 2, 0)), buffer.pop());
        assert_eq!(Some(TestTask::new(7, 3, 0)), buffer.pop());
        assert_eq!(None, buffer.pop());
    }

    #[test]
    fn test_task_with_duration_not_ready_not_popped() {
        set_fake_time(0);
        let mut buffer = ThreadBuffer::<TestTask>::new(8);

        buffer.push(TestTask::new(1, 10, 100));

        // Task not ready yet
        assert_eq!(None, buffer.pop());

        // Task still not ready
        set_fake_time(50);
        assert_eq!(None, buffer.pop());
    }

    #[test]
    fn test_task_with_duration_becomes_ready() {
        set_fake_time(0);
        let mut buffer = ThreadBuffer::<TestTask>::new(8);

        let task = TestTask::new(1, 10, 100);
        buffer.push(task.clone());

        // Task not ready yet at time 0
        assert_eq!(None, buffer.pop());

        // Time passes, task becomes ready at time 100
        set_fake_time(100);
        let popped = buffer.pop();
        assert!(popped.is_some());
        let popped_task = popped.unwrap();
        assert_eq!(popped_task.stream_id, 1);
        assert_eq!(popped_task.value, 10);
        assert_eq!(popped_task.duration, 100);

        assert_eq!(None, buffer.pop());
    }

    #[test]
    fn test_multiple_tasks_with_different_durations() {
        set_fake_time(0);
        let mut buffer = ThreadBuffer::<TestTask>::new(8);

        // Push tasks on different streams with different durations
        // They will be tried in push order when popping
        buffer.push(TestTask::new(1, 100, 0));
        buffer.push(TestTask::new(2, 200, 50));
        buffer.push(TestTask::new(3, 300, 100));

        // At time 0, stream 1 is ready, so it gets popped
        let task1 = buffer.pop();
        assert!(task1.is_some());
        let t1 = task1.unwrap();
        assert_eq!(t1.stream_id, 1);
        assert_eq!(t1.value, 100);

        // At time 50, stream 2 is ready (stream 1 is gone, stream 2 is now front)
        set_fake_time(50);
        let task2 = buffer.pop();
        assert!(task2.is_some());
        let t2 = task2.unwrap();
        assert_eq!(t2.stream_id, 2);
        assert_eq!(t2.value, 200);

        // At time 100, stream 3 is ready
        set_fake_time(100);
        let task3 = buffer.pop();
        assert!(task3.is_some());
        let t3 = task3.unwrap();
        assert_eq!(t3.stream_id, 3);
        assert_eq!(t3.value, 300);
        assert_eq!(None, buffer.pop());
    }

    #[test]
    fn test_multiple_tasks_same_stream_with_duration() {
        set_fake_time(0);
        let mut buffer = ThreadBuffer::<TestTask>::new(8);

        // Push multiple tasks on the same stream with different durations
        buffer.push(TestTask::new(1, 10, 50));
        buffer.push(TestTask::new(1, 11, 50)); // Same stream, same duration
        buffer.push(TestTask::new(1, 12, 50));

        // At time 50, all tasks should be ready in FIFO order
        set_fake_time(50);
        let t1 = buffer.pop().unwrap();
        assert_eq!(t1.stream_id, 1);
        assert_eq!(t1.value, 10);

        let t2 = buffer.pop().unwrap();
        assert_eq!(t2.stream_id, 1);
        assert_eq!(t2.value, 11);

        let t3 = buffer.pop().unwrap();
        assert_eq!(t3.stream_id, 1);
        assert_eq!(t3.value, 12);

        assert_eq!(None, buffer.pop());
    }

    #[test]
    fn test_skip_not_ready_tasks_in_same_stream() {
        set_fake_time(0);
        let mut buffer = ThreadBuffer::<TestTask>::new(8);

        // Push tasks with different durations on same stream
        // Since both tasks are on the same stream, if the front one isn't ready,
        // we can't access the next one without popping the first
        buffer.push(TestTask::new(1, 10, 50));
        buffer.push(TestTask::new(1, 11, 100));

        // At time 50, first task is ready and should be popped
        set_fake_time(50);
        let task = buffer.pop();
        assert!(task.is_some());
        let popped = task.unwrap();
        assert_eq!(popped.stream_id, 1);
        assert_eq!(popped.value, 10);

        // Now second task is ready at time 100
        set_fake_time(100);
        let task = buffer.pop();
        assert!(task.is_some());
        let popped = task.unwrap();
        assert_eq!(popped.stream_id, 1);
        assert_eq!(popped.value, 11);
    }

    #[test]
    fn test_mixed_ready_and_not_ready_streams() {
        set_fake_time(0);
        let mut buffer = ThreadBuffer::<TestTask>::new(8);

        // Create multiple streams all with immediate readiness
        buffer.push(TestTask::new(1, 10, 0));
        buffer.push(TestTask::new(2, 20, 0));
        buffer.push(TestTask::new(3, 30, 0));
        buffer.push(TestTask::new(4, 40, 0));

        // All tasks are ready immediately, should pop in FIFO order
        let t1 = buffer.pop().unwrap();
        assert_eq!(t1.stream_id, 1);
        assert_eq!(t1.value, 10);

        let t2 = buffer.pop().unwrap();
        assert_eq!(t2.stream_id, 2);
        assert_eq!(t2.value, 20);

        let t3 = buffer.pop().unwrap();
        assert_eq!(t3.stream_id, 3);
        assert_eq!(t3.value, 30);

        let t4 = buffer.pop().unwrap();
        assert_eq!(t4.stream_id, 4);
        assert_eq!(t4.value, 40);

        assert_eq!(None, buffer.pop());
    }
}
