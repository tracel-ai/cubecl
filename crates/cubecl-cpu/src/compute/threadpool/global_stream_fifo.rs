use std::collections::VecDeque;

use crate::compute::threadpool::thread_buffer::GetId;

pub struct GlobalStreamFifo<T: GetId> {
    buffer: VecDeque<T>,
    id: usize,
}

impl<T: GetId> GlobalStreamFifo<T> {
    pub fn new(id: usize) -> Self {
        Self {
            buffer: VecDeque::new(),
            id,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn is_same_id(&self, elem: &T) -> bool {
        self.id == elem.get_id()
    }

    pub fn pop(&mut self) -> Option<T> {
        self.buffer.pop_front()
    }

    pub fn push(&mut self, elem: T) {
        self.buffer.push_back(elem);
    }
}
