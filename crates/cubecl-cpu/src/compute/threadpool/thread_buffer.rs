use std::sync::Arc;

use crate::compute::threadpool::global_stream_fifo::GlobalStreamFifo;

use super::thread_stream_fifo::ThreadStreamFifo;

pub trait GetId {
    fn get_id(&self) -> usize;
}

pub struct ThreadBuffer<T: GetId, const NB_STREAM: usize, const CAPACITY: usize> {
    fifos: [ThreadStreamFifo<T, CAPACITY>; NB_STREAM],
    global: Arc<spin::Mutex<GlobalStreamFifo<T>>>,
    threads_buffer: Arc<[spin::Mutex<ThreadBuffer<T, NB_STREAM, CAPACITY>>]>,
    front: usize,
    len: usize,
    index: usize,
}

impl<T: GetId, const NB_STREAM: usize, const CAPACITY: usize> ThreadBuffer<T, NB_STREAM, CAPACITY> {
    /// Push to the local queue if there's still place
    fn try_push_local(&mut self, elem: T) -> Option<T> {
        for i in self.front..(self.front + self.len) {
            let i = i % CAPACITY;
            if self.fifos[i].is_same_id(&elem) {
                if self.fifos[i].is_full() {
                    return Some(elem);
                }
                self.fifos[i].push(elem);
                return None;
            }
        }
        if self.len >= CAPACITY {
            return Some(elem);
        }
        self.fifos[(self.front + self.len) % CAPACITY].push(elem);
        self.len += 1;
        None
    }

    /// Push to the LocalBuffer if local capacity allow automatically overflow to the global queue
    pub fn push(&mut self, elem: T) {
        let Some(elem_not_accepted_locally) = self.try_push_local(elem) else {
            return;
        };
        self.global.lock().push(elem_not_accepted_locally);
    }

    fn pop_local(&mut self) -> Option<T> {
        let local = self.fifos.iter_mut().find(|fifo| !fifo.is_empty())?;
        if local.len() == 1 {
            self.front = (self.front + 1) % CAPACITY;
        }
        local.pop()
    }

    fn try_steal(&mut self) -> Option<&mut ThreadStreamFifo<T, CAPACITY>> {
        if self.fifos.len() <= 1 {
            return None;
        }

        return self.fifos.get_mut((self.front + self.len) % CAPACITY);
    }

    pub fn pop(&mut self) -> Option<T> {
        if let Some(value) = self.pop_local() {
            return Some(value);
        }
        if let Some(value) = self.global.lock().pop() {
            return Some(value);
        }
        for (i, thread_buffer) in self.threads_buffer.iter().enumerate() {
            if i == self.index {
                continue;
            }
            if let Some(fifo) = thread_buffer.lock().try_steal() {
                if fifo.len() == 1 {
                    return fifo.pop();
                }
                fifo.drain(&mut self.fifos[self.front % CAPACITY]);
                self.len += 1;
                return self.fifos[self.front].pop();
            }
        }
        None
    }
}
