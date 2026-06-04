use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};

use crate::compute::threadpool::circular_buffer::CircularBuffer;

pub trait GetId {
    fn get_id(&self) -> usize;
}

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

    fn push_new(&mut self, elem: T) {
        let id = elem.get_id();
        let mut fifo = match self.empty_streams.pop() {
            Some(fifo) => fifo,
            None => VecDeque::new(),
        };
        fifo.push_back(elem);
        self.streams.push(fifo);
        self.streams_id.entry(id).insert_entry(self.streams.back());
    }

    pub fn push(&mut self, elem: T) {
        match self.streams_id.get(&elem.get_id()) {
            Some(&i) => {
                self.streams[i].push_back(elem);
            }
            None => self.push_new(elem),
        }
    }
}
