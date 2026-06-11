use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};

use super::circular_buffer::CircularBuffer;

pub trait GetId {
    fn get_id(&self) -> usize;
    fn is_ready(&self) -> bool;
}

/// Local thread buffer used for pushing task to a worker for the scheduler and for getting/stealing task for the worker
pub struct ThreadBuffer<T: GetId> {
    /// Main stream storage
    streams: CircularBuffer<VecDeque<T>>,
    /// Empty streams available to be reused to avoid reallocation
    empty_streams: Vec<VecDeque<T>>,
    /// Association of stream id to position in `streams`
    streams_id: HashMap<usize, usize>,
    /// Reference to other thread buffer to be able to steal
    threads_buffer: Arc<[spin::Mutex<ThreadBuffer<T>>]>,
    /// Current thread id
    thread_id: usize,
}

impl<T: GetId> ThreadBuffer<T> {
    /// Construct a ThreadBuffer with an empty reference to all thread_buffer
    pub fn new(thread_id: usize, capacity: usize) -> Self {
        let streams = CircularBuffer::new(capacity);
        let streams_id = HashMap::new();
        let empty_streams = Vec::new();
        Self {
            streams,
            empty_streams,
            streams_id,
            threads_buffer: Arc::new([]),
            thread_id,
        }
    }

    pub fn set_threads_buffer(&mut self, threads_buffer: Arc<[spin::Mutex<ThreadBuffer<T>>]>) {
        self.threads_buffer = threads_buffer;
    }

    /// This function assume that the fifo is not empty
    fn push_fifo(&mut self, fifo: VecDeque<T>) {
        let id = fifo.front().unwrap().get_id();
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
    pub fn push(&mut self, elem: T) {
        match self.streams_id.get(&elem.get_id()) {
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
                self.streams.pop_front();
                continue;
            }
            if !fifos.front().unwrap().is_ready() {
                continue;
            }
            let elem = fifos.pop_front().unwrap();
            if fifos.is_empty() {
                let stream = self.streams.pop_front().unwrap();
                self.empty_streams.push(stream);
                self.streams_id.remove(&elem.get_id());
            }
            return Some(elem);
        }
        return None;
    }

    fn steal(&mut self) -> Option<T> {
        for i in 0..self.threads_buffer.len() {
            if self.thread_id == i {
                continue;
            }
            let mut thread = self.threads_buffer[i].lock();
            if thread.streams.len() <= 1 {
                continue;
            }
            let mut last_stream = thread.streams.pop_back().unwrap();
            let stream_id = last_stream.front().unwrap().get_id();

            thread.streams_id.remove(&stream_id);
            drop(thread);
            if !last_stream.front().unwrap().is_ready() {
                return None;
            }
            let elem = last_stream.pop_front().unwrap();
            if last_stream.is_empty() {
                self.empty_streams.push(last_stream);
            } else {
                self.push_fifo(last_stream);
            }
            return Some(elem);
        }
        None
    }

    /// Pop task for local execution
    pub fn pop(&mut self) -> Option<T> {
        if let Some(elem) = self.pop_local() {
            return Some(elem);
        }
        if self.streams.is_empty() {
            if let Some(elem) = self.steal() {
                return Some(elem);
            }
        }
        None
    }

    /// Pop task with id for flushing
    pub fn pop_id(&mut self, thread_id: usize) -> Option<T> {
        let id = self.streams_id.get(&thread_id)?;
        self.empty_streams[*id].pop_front()
    }
}

#[cfg(test)]
mod tests {
    use super::{GetId, ThreadBuffer};
    use std::cell::RefCell;
    use std::sync::Arc;

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

    impl GetId for TestTask {
        fn get_id(&self) -> usize {
            self.stream_id
        }

        fn is_ready(&self) -> bool {
            let elapsed = get_fake_time().saturating_sub(self.created_at);
            elapsed >= self.duration
        }
    }

    fn setup_buffers(
        thread_count: usize,
        capacity: usize,
    ) -> Arc<[spin::Mutex<ThreadBuffer<TestTask>>]> {
        let buffers = (0..thread_count)
            .map(|thread_id| spin::Mutex::new(ThreadBuffer::new(thread_id, capacity)))
            .collect::<Vec<_>>();
        let shared: Arc<[spin::Mutex<ThreadBuffer<TestTask>>]> = buffers.into();
        for buffer in shared.iter() {
            buffer.lock().set_threads_buffer(shared.clone());
        }
        shared
    }

    #[test]
    fn test_pop_empty_returns_none() {
        let mut buffer = ThreadBuffer::<TestTask>::new(0, 8);
        assert_eq!(None, buffer.pop());
        assert_eq!(None, buffer.pop());
    }

    #[test]
    fn test_single_stream_fifo_pop_order() {
        set_fake_time(0);
        let mut buffer = ThreadBuffer::<TestTask>::new(0, 8);

        buffer.push(TestTask::new(1, 10, 0));
        buffer.push(TestTask::new(1, 11, 0));
        buffer.push(TestTask::new(1, 12, 0));

        assert_eq!(Some(TestTask::new(1, 10, 0)), buffer.pop());
        assert_eq!(Some(TestTask::new(1, 11, 0)), buffer.pop());
        assert_eq!(Some(TestTask::new(1, 12, 0)), buffer.pop());
        assert_eq!(None, buffer.pop());
    }

    #[test]
    fn test_steal_requires_source_with_more_than_one_stream() {
        set_fake_time(0);
        let shared = setup_buffers(2, 8);

        {
            let mut source = shared[0].lock();
            source.push(TestTask::new(10, 1, 0));
        }

        let stolen = shared[1].lock().pop();
        assert_eq!(None, stolen);

        {
            let mut source = shared[0].lock();
            source.push(TestTask::new(20, 2, 0));
        }

        let stolen = shared[1].lock().pop();
        assert_eq!(Some(TestTask::new(20, 2, 0)), stolen);

        let source_remaining = shared[0].lock().pop();
        assert_eq!(Some(TestTask::new(10, 1, 0)), source_remaining);
    }

    #[test]
    fn test_steal_from_other_thread_returns_work() {
        set_fake_time(0);
        let shared = setup_buffers(2, 8);

        {
            let mut source = shared[0].lock();
            source.push(TestTask::new(10, 100, 0));
            source.push(TestTask::new(20, 200, 0));
            source.push(TestTask::new(20, 201, 0));
        }

        let stolen = shared[1].lock().pop();
        assert_eq!(Some(TestTask::new(20, 200, 0)), stolen);

        let thief_next = shared[1].lock().pop();
        assert_eq!(Some(TestTask::new(20, 201, 0)), thief_next);

        let source_next = shared[0].lock().pop();
        assert_eq!(Some(TestTask::new(10, 100, 0)), source_next);
    }

    #[test]
    fn test_repush_same_stream_after_progress() {
        set_fake_time(0);
        let mut buffer = ThreadBuffer::<TestTask>::new(0, 8);

        buffer.push(TestTask::new(7, 1, 0));
        buffer.push(TestTask::new(7, 2, 0));
        assert_eq!(Some(TestTask::new(7, 1, 0)), buffer.pop());

        buffer.push(TestTask::new(7, 3, 0));

        assert_eq!(Some(TestTask::new(7, 2, 0)), buffer.pop());
        assert_eq!(Some(TestTask::new(7, 3, 0)), buffer.pop());
        assert_eq!(None, buffer.pop());
    }

    #[test]
    fn test_steal_then_push_same_stream_on_thief() {
        set_fake_time(0);
        let shared = setup_buffers(2, 8);

        {
            let mut source = shared[0].lock();
            source.push(TestTask::new(1, 1, 0));
            source.push(TestTask::new(2, 10, 0));
            source.push(TestTask::new(2, 11, 0));
        }

        let first = shared[1].lock().pop();
        assert_eq!(Some(TestTask::new(2, 10, 0)), first);

        {
            let mut thief = shared[1].lock();
            thief.push(TestTask::new(2, 12, 0));
        }

        let second = shared[1].lock().pop();
        let third = shared[1].lock().pop();
        assert_eq!(Some(TestTask::new(2, 11, 0)), second);
        assert_eq!(Some(TestTask::new(2, 12, 0)), third);
        assert_eq!(None, shared[1].lock().pop());
    }

    #[test]
    fn test_task_with_duration_not_ready_not_popped() {
        set_fake_time(0);
        let mut buffer = ThreadBuffer::<TestTask>::new(0, 8);

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
        let mut buffer = ThreadBuffer::<TestTask>::new(0, 8);

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
        let mut buffer = ThreadBuffer::<TestTask>::new(0, 8);

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
        let mut buffer = ThreadBuffer::<TestTask>::new(0, 8);

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
        let mut buffer = ThreadBuffer::<TestTask>::new(0, 8);

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
    fn test_steal_respects_ready_status() {
        set_fake_time(0);
        let shared = setup_buffers(2, 8);

        {
            let mut source = shared[0].lock();
            // Add two immediately ready streams
            source.push(TestTask::new(10, 100, 0)); // Ready immediately
            source.push(TestTask::new(20, 200, 0)); // Ready immediately
        }

        // Thread 1 can steal the last stream from thread 0 (has >1 streams)
        let stolen = shared[1].lock().pop();
        assert!(stolen.is_some());
        let task = stolen.unwrap();
        assert_eq!(task.stream_id, 20);
        assert_eq!(task.value, 200);

        // Thread 0 can now pop its remaining stream
        let remaining = shared[0].lock().pop();
        assert!(remaining.is_some());
        let task = remaining.unwrap();
        assert_eq!(task.stream_id, 10);
        assert_eq!(task.value, 100);
    }

    #[test]
    fn test_mixed_ready_and_not_ready_streams() {
        set_fake_time(0);
        let mut buffer = ThreadBuffer::<TestTask>::new(0, 8);

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
