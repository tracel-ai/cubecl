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
        loop {
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
    use std::sync::Arc;

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct TestTask {
        stream_id: usize,
        value: usize,
    }

    impl TestTask {
        fn new(stream_id: usize, value: usize) -> Self {
            Self { stream_id, value }
        }
    }

    impl GetId for TestTask {
        fn get_id(&self) -> usize {
            self.stream_id
        }
        fn is_ready(&self) -> bool {
            true
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
        let mut buffer = ThreadBuffer::<TestTask>::new(0, 8);

        buffer.push(TestTask::new(1, 10));
        buffer.push(TestTask::new(1, 11));
        buffer.push(TestTask::new(1, 12));

        assert_eq!(Some(TestTask::new(1, 10)), buffer.pop());
        assert_eq!(Some(TestTask::new(1, 11)), buffer.pop());
        assert_eq!(Some(TestTask::new(1, 12)), buffer.pop());
        assert_eq!(None, buffer.pop());
    }

    #[test]
    fn test_steal_requires_source_with_more_than_one_stream() {
        let shared = setup_buffers(2, 8);

        {
            let mut source = shared[0].lock();
            source.push(TestTask::new(10, 1));
        }

        let stolen = shared[1].lock().pop();
        assert_eq!(None, stolen);

        {
            let mut source = shared[0].lock();
            source.push(TestTask::new(20, 2));
        }

        let stolen = shared[1].lock().pop();
        assert_eq!(Some(TestTask::new(20, 2)), stolen);

        let source_remaining = shared[0].lock().pop();
        assert_eq!(Some(TestTask::new(10, 1)), source_remaining);
    }

    #[test]
    fn test_steal_from_other_thread_returns_work() {
        let shared = setup_buffers(2, 8);

        {
            let mut source = shared[0].lock();
            source.push(TestTask::new(10, 100));
            source.push(TestTask::new(20, 200));
            source.push(TestTask::new(20, 201));
        }

        let stolen = shared[1].lock().pop();
        assert_eq!(Some(TestTask::new(20, 200)), stolen);

        let thief_next = shared[1].lock().pop();
        assert_eq!(Some(TestTask::new(20, 201)), thief_next);

        let source_next = shared[0].lock().pop();
        assert_eq!(Some(TestTask::new(10, 100)), source_next);
    }

    #[test]
    fn test_repush_same_stream_after_progress() {
        let mut buffer = ThreadBuffer::<TestTask>::new(0, 8);

        buffer.push(TestTask::new(7, 1));
        buffer.push(TestTask::new(7, 2));
        assert_eq!(Some(TestTask::new(7, 1)), buffer.pop());

        buffer.push(TestTask::new(7, 3));

        assert_eq!(Some(TestTask::new(7, 2)), buffer.pop());
        assert_eq!(Some(TestTask::new(7, 3)), buffer.pop());
        assert_eq!(None, buffer.pop());
    }

    #[test]
    fn test_steal_then_push_same_stream_on_thief() {
        let shared = setup_buffers(2, 8);

        {
            let mut source = shared[0].lock();
            source.push(TestTask::new(1, 1));
            source.push(TestTask::new(2, 10));
            source.push(TestTask::new(2, 11));
        }

        let first = shared[1].lock().pop();
        assert_eq!(Some(TestTask::new(2, 10)), first);

        {
            let mut thief = shared[1].lock();
            thief.push(TestTask::new(2, 12));
        }

        let second = shared[1].lock().pop();
        let third = shared[1].lock().pop();
        assert_eq!(Some(TestTask::new(2, 11)), second);
        assert_eq!(Some(TestTask::new(2, 12)), third);
        assert_eq!(None, shared[1].lock().pop());
    }
}
