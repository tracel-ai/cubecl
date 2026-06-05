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
        self.streams.push_back(fifo);
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

    fn pop_local(&mut self) -> Option<T> {
        if self.streams.is_empty() {
            return None;
        }
        let front = self.streams.front();
        let fifos = &mut self.streams[front];
        let elem = fifos.pop_back().unwrap();
        if fifos.is_empty() {
            let stream = self.streams.pop_front().unwrap();
            self.empty_streams.push(stream);
            self.streams_id.remove(&elem.get_id());
        }
        Some(elem)
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
            drop(thread);
            let elem = last_stream.pop_front().unwrap();
            if last_stream.is_empty() {
                self.empty_streams.push(last_stream);
            } else {
                self.streams.push_back(last_stream);
            }
            return Some(elem);
        }
        None
    }

    pub fn pop(&mut self) -> Option<T> {
        if let Some(elem) = self.pop_local() {
            return Some(elem);
        }
        if let Some(elem) = self.steal() {
            return Some(elem);
        }
        None
    }
}
